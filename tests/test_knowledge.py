"""Embeddings and knowledge-base retrieval on both backends.

The pgvector variants run only when TEST_POSTGRES_URL points at a server with the `vector`
extension (GitHub Actions CI uses the pgvector/pgvector image), e.g.
    TEST_POSTGRES_URL=postgresql+psycopg://support:support@localhost:5432/support uv run pytest tests/test_knowledge.py
"""

from __future__ import annotations

import math
import os
import uuid
from collections.abc import AsyncIterator

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from support_agent.config import Settings
from support_agent.knowledge import (
    EmbeddingModel,
    HashingEmbeddings,
    InMemoryKnowledgeBase,
    KnowledgeBase,
    build_embedding_model,
    build_knowledge_base,
    hashing_model,
)
from support_agent.knowledge.embeddings import tokenize
from support_agent.rules.policy import PolicyConfig
from support_agent.store_api import StoreRepository
from support_agent.store_api.schemas import KnowledgeArticle

POSTGRES_URL = os.environ.get("TEST_POSTGRES_URL")
ARTICLES = StoreRepository().list_knowledge_articles()


class CountingEmbeddings(HashingEmbeddings):
    """Records which texts get embedded, to verify incremental syncs."""

    def __init__(self, dimensions: int = 1536) -> None:
        super().__init__(dimensions)
        self.embedded: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.embedded.extend(texts)
        return super().embed_documents(texts)


# --- Embeddings ------------------------------------------------------------------------------


def test_tokenizer_stems_and_drops_stopwords() -> None:
    assert tokenize("Do you ship? Shipping, shipped, ships!") == ["ship", "ship", "ship", "ship"]
    assert tokenize("charges and the charge") == ["charg", "charg"]


def test_hashing_embeddings_are_deterministic_unit_vectors() -> None:
    first, second = HashingEmbeddings(), HashingEmbeddings()
    vector = first.embed_query("Do you ship to Canada?")
    assert vector == second.embed_query("Do you ship to Canada?")
    assert len(vector) == 1536
    assert math.isclose(sum(x * x for x in vector), 1.0)
    assert not any(first.embed_query("the and of"))


def test_openai_embeddings_are_configured_from_settings() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        embeddings_provider="openai",
        openai_api_key="sk-test",
        embeddings_dimensions=256,
    )
    built = build_embedding_model(settings)
    assert (built.name, built.dimensions) == ("openai/text-embedding-3-small", 256)
    assert built.embeddings.dimensions == 256  # type: ignore[attr-defined]


def test_pgvector_backend_requires_postgres(settings: Settings) -> None:
    engine = create_async_engine(settings.database_url)
    with pytest.raises(RuntimeError, match="PostgreSQL"):
        build_knowledge_base(settings.model_copy(update={"kb_backend": "pgvector"}), engine)


# --- Retrieval on both backends --------------------------------------------------------------


@pytest.fixture(params=["memory", "pgvector"])
async def knowledge_base(request: pytest.FixtureRequest) -> AsyncIterator[KnowledgeBase]:
    if request.param == "memory":
        memory = InMemoryKnowledgeBase(hashing_model())
        await memory.index(ARTICLES)
        yield memory
        return
    if not POSTGRES_URL:
        pytest.skip("TEST_POSTGRES_URL not set")
    from support_agent.knowledge.pgvector import PgVectorKnowledgeBase

    engine = create_async_engine(POSTGRES_URL)
    table = f"kb_test_{uuid.uuid4().hex[:8]}"
    postgres = PgVectorKnowledgeBase(engine, hashing_model(), table_name=table)
    await postgres.index(ARTICLES)
    try:
        yield postgres
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        await engine.dispose()


@pytest.mark.parametrize(
    ("query", "expected_top"),
    [
        ("Do you ship to Canada? How long does international shipping take?", "KB-001"),
        ("how long until my refund shows up on my card", "KB-004"),
        ("warranty on my espresso machine", "KB-005"),
        ("how do I clean the burr grinder", "KB-008"),
        ("How do I pause my monthly bean subscription?", "KB-011"),
        ("Can I use two discount codes on one order?", "KB-013"),
    ],
)
async def test_search_ranks_relevant_article_first(
    knowledge_base: KnowledgeBase, query: str, expected_top: str
) -> None:
    hits = await knowledge_base.search(query)
    assert hits[0].id == expected_top
    assert len(hits) <= 3
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert all(0 < h.score <= 1 for h in hits)


async def test_search_returns_full_articles_with_snippets(knowledge_base: KnowledgeBase) -> None:
    top = (await knowledge_base.search("Do gift cards expire?"))[0]
    assert (top.id, top.title) == ("KB-010", "Gift cards")
    assert top.content.startswith("Digital gift cards")
    assert top.snippet == top.content  # short article: the snippet is the whole text


async def test_unrelated_query_stays_below_the_answer_threshold(knowledge_base: KnowledgeBase) -> None:
    hits = await knowledge_base.search("Are you hiring baristas in Portland?")
    assert all(h.score < PolicyConfig().kb_min_score for h in hits)


async def test_stopword_only_query_returns_nothing(knowledge_base: KnowledgeBase) -> None:
    assert await knowledge_base.search("what is it") == []


async def test_reindex_replaces_the_article_set() -> None:
    knowledge_base = InMemoryKnowledgeBase(hashing_model())
    await knowledge_base.index(ARTICLES)
    await knowledge_base.index([a for a in ARTICLES if a.id != "KB-010"])
    assert all(h.id != "KB-010" for h in await knowledge_base.search("Do gift cards expire?"))


# --- pgvector specifics ----------------------------------------------------------------------


@pytest.mark.skipif(not POSTGRES_URL, reason="TEST_POSTGRES_URL not set")
async def test_pgvector_sync_is_incremental_and_rebuilds_on_dimension_change() -> None:
    from support_agent.knowledge.pgvector import PgVectorKnowledgeBase

    engine = create_async_engine(POSTGRES_URL or "")
    table = f"kb_test_{uuid.uuid4().hex[:8]}"
    embeddings = CountingEmbeddings()
    model = EmbeddingModel(embeddings, "counting", 1536)
    try:
        await PgVectorKnowledgeBase(engine, model, table_name=table).index(ARTICLES)
        assert len(embeddings.embedded) == len(ARTICLES)

        # Unchanged articles are not embedded again; an edited one is; a removed one disappears.
        embeddings.embedded.clear()
        edited = ARTICLES[0].model_copy(update={"content": "We ship worldwide except Antarctica."})
        remaining: list[KnowledgeArticle] = [edited, *ARTICLES[1:-1]]
        knowledge_base = PgVectorKnowledgeBase(engine, model, table_name=table)
        await knowledge_base.index(remaining)
        assert embeddings.embedded == ["International shipping\n\nWe ship worldwide except Antarctica."]
        assert (await knowledge_base.search("ship worldwide Antarctica"))[0].id == "KB-001"
        async with engine.connect() as conn:
            assert await conn.scalar(text(f"SELECT count(*) FROM {table}")) == len(ARTICLES) - 1

        # A different embedding size rebuilds the table instead of failing on the old column type.
        smaller = EmbeddingModel(CountingEmbeddings(256), "counting", 256)
        resized = PgVectorKnowledgeBase(engine, smaller, table_name=table)
        await resized.index(remaining)
        assert (await resized.search("Do gift cards expire?"))[0].id == "KB-010"
    finally:
        async with engine.begin() as conn:
            await conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
        await engine.dispose()
