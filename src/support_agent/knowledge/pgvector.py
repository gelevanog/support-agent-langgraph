"""PostgreSQL + pgvector backend: vectors live next to tickets and checkpoints in the same database.

`index()` is an incremental sync: only new or edited articles (or vectors from a different
embedding model) are embedded, removed articles are deleted, and a change of dimensions rebuilds
the table. Search uses an HNSW index with cosine distance. Requires the `postgres` extra and a
server with the `vector` extension (e.g. the `pgvector/pgvector` Docker image).
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

from pgvector.sqlalchemy import VECTOR
from sqlalchemy import Column, Index, MetaData, String, Table, Text, delete, select, text
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine

from support_agent.knowledge.base import KnowledgeBase, article_text
from support_agent.knowledge.embeddings import EmbeddingModel
from support_agent.logging_config import get_logger
from support_agent.models import RetrievedArticle
from support_agent.store_api.schemas import KnowledgeArticle

log = get_logger(__name__)

# Serialises index syncs when several app instances start at the same time.
_SYNC_LOCK_ID = 0x5A_4B_0001


def _content_hash(article: KnowledgeArticle) -> str:
    return hashlib.sha256(article_text(article).encode()).hexdigest()


class PgVectorKnowledgeBase(KnowledgeBase):
    backend = "pgvector"

    def __init__(self, engine: AsyncEngine, model: EmbeddingModel, table_name: str = "kb_article_embeddings") -> None:
        super().__init__(model)
        self._engine = engine
        self._metadata = MetaData()
        self._table = Table(
            table_name,
            self._metadata,
            Column("id", String(64), primary_key=True),
            Column("title", Text, nullable=False),
            Column("content", Text, nullable=False),
            Column("content_hash", String(64), nullable=False),
            Column("embedding_model", String(128), nullable=False),
            Column("embedding", VECTOR(model.dimensions), nullable=False),
            Index(
                f"ix_{table_name}_embedding",
                "embedding",
                postgresql_using="hnsw",
                postgresql_ops={"embedding": "vector_cosine_ops"},
            ),
        )

    async def _drop_if_dimensions_changed(self, conn: AsyncConnection) -> None:
        current = await conn.scalar(
            text(
                "SELECT format_type(atttypid, atttypmod) FROM pg_attribute "
                "WHERE attrelid = to_regclass(:table) AND attname = 'embedding'"
            ),
            {"table": self._table.name},
        )
        if current is not None and current != f"vector({self.model.dimensions})":
            log.warning("kb.rebuild", reason=f"dimensions changed from {current} to {self.model.dimensions}")
            await conn.run_sync(self._table.drop)

    async def index(self, articles: Sequence[KnowledgeArticle]) -> None:
        table = self._table
        async with self._engine.begin() as conn:
            await conn.execute(text("SELECT pg_advisory_xact_lock(:id)"), {"id": _SYNC_LOCK_ID})
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            await self._drop_if_dimensions_changed(conn)
            await conn.run_sync(self._metadata.create_all)

            rows = await conn.execute(select(table.c.id, table.c.content_hash, table.c.embedding_model))
            stored = {row.id: (row.content_hash, row.embedding_model) for row in rows}
            stale = [a for a in articles if stored.get(a.id) != (_content_hash(a), self.model.name)]
            if stale:
                vectors = await self.model.embeddings.aembed_documents([article_text(a) for a in stale])
                upsert = insert(table)
                await conn.execute(
                    upsert.on_conflict_do_update(
                        index_elements=[table.c.id],
                        set_={
                            name: upsert.excluded[name]
                            for name in ("title", "content", "content_hash", "embedding_model", "embedding")
                        },
                    ),
                    [
                        {
                            "id": article.id,
                            "title": article.title,
                            "content": article.content,
                            "content_hash": _content_hash(article),
                            "embedding_model": self.model.name,
                            "embedding": vector,
                        }
                        for article, vector in zip(stale, vectors, strict=True)
                    ],
                )
            removed = await conn.execute(delete(table).where(table.c.id.not_in([a.id for a in articles])))
        log.info(
            "kb.indexed",
            backend=self.backend,
            model=self.model.name,
            articles=len(articles),
            embedded=len(stale),
            removed=removed.rowcount,
        )

    async def _nearest(self, vector: list[float], limit: int) -> list[RetrievedArticle]:
        table = self._table
        distance = table.c.embedding.cosine_distance(vector)
        query = (
            select(table.c.id, table.c.title, table.c.content, (1 - distance).label("score"))
            .order_by(distance)
            .limit(limit)
        )
        async with self._engine.connect() as conn:
            rows = await conn.execute(query)
            return [
                RetrievedArticle(id=row.id, title=row.title, content=row.content, score=round(row.score, 3))
                for row in rows
            ]
