"""In-memory vector index: exact cosine search, zero infrastructure, rebuilt on every start."""

from __future__ import annotations

from collections.abc import Sequence

from support_agent.knowledge.base import KnowledgeBase, article_text
from support_agent.knowledge.embeddings import EmbeddingModel, normalize
from support_agent.logging_config import get_logger
from support_agent.models import RetrievedArticle
from support_agent.store_api.schemas import KnowledgeArticle

log = get_logger(__name__)


class InMemoryKnowledgeBase(KnowledgeBase):
    """Brute-force search over unit vectors. Fine for help centers up to a few thousand articles."""

    backend = "memory"

    def __init__(self, model: EmbeddingModel) -> None:
        super().__init__(model)
        self._entries: list[tuple[KnowledgeArticle, list[float]]] = []

    async def index(self, articles: Sequence[KnowledgeArticle]) -> None:
        vectors = await self.model.embeddings.aembed_documents([article_text(a) for a in articles]) if articles else []
        self._entries = [(article, normalize(vector)) for article, vector in zip(articles, vectors, strict=True)]
        log.info("kb.indexed", backend=self.backend, model=self.model.name, articles=len(self._entries))

    async def _nearest(self, vector: list[float], limit: int) -> list[RetrievedArticle]:
        query = normalize(vector)
        scored = [(sum(q * d for q, d in zip(query, doc, strict=True)), article) for article, doc in self._entries]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [RetrievedArticle(**article.model_dump(), score=round(score, 3)) for score, article in scored[:limit]]
