"""Knowledge-base retrieval: embedding models and vector-search backends (in-memory, pgvector)."""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncEngine

from support_agent.config import Settings
from support_agent.knowledge.base import KnowledgeBase
from support_agent.knowledge.embeddings import EmbeddingModel, HashingEmbeddings, build_embedding_model, hashing_model
from support_agent.knowledge.memory import InMemoryKnowledgeBase


def build_knowledge_base(settings: Settings, engine: AsyncEngine) -> KnowledgeBase:
    """`KB_BACKEND=memory` (default) or `pgvector` (stores vectors in the Postgres `DATABASE_URL`)."""
    model = build_embedding_model(settings)
    match settings.kb_backend:
        case "memory":
            return InMemoryKnowledgeBase(model)
        case "pgvector":
            if engine.dialect.name != "postgresql":
                raise RuntimeError("KB_BACKEND=pgvector requires a PostgreSQL DATABASE_URL")
            try:
                from support_agent.knowledge.pgvector import PgVectorKnowledgeBase
            except ImportError as exc:  # pragma: no cover - depends on installed extras
                raise RuntimeError("pgvector support requires: uv sync --extra postgres") from exc
            return PgVectorKnowledgeBase(engine, model)


__all__ = [
    "EmbeddingModel",
    "HashingEmbeddings",
    "InMemoryKnowledgeBase",
    "KnowledgeBase",
    "build_embedding_model",
    "build_knowledge_base",
    "hashing_model",
]
