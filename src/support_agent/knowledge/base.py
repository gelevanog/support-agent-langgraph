"""The retriever interface the agent depends on."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from support_agent.knowledge.embeddings import EmbeddingModel
from support_agent.models import RetrievedArticle
from support_agent.store_api.schemas import KnowledgeArticle


def article_text(article: KnowledgeArticle) -> str:
    """The text that gets embedded for an article."""
    return f"{article.title}\n\n{article.content}"


class KnowledgeBase(ABC):
    """Semantic search over help-center articles.

    Backends only implement indexing and nearest-neighbour lookup; embedding the query and the
    result contract (best first, cosine score in (0, 1], nothing for an empty query) live here.
    """

    backend: str

    def __init__(self, model: EmbeddingModel) -> None:
        self.model = model

    @abstractmethod
    async def index(self, articles: Sequence[KnowledgeArticle]) -> None:
        """Sync the index so it contains exactly `articles`."""

    @abstractmethod
    async def _nearest(self, vector: list[float], limit: int) -> list[RetrievedArticle]:
        """The `limit` articles closest to `vector` by cosine similarity, best first."""

    async def search(self, query: str, limit: int = 3) -> list[RetrievedArticle]:
        vector = await self.model.embeddings.aembed_query(query)
        if not any(vector):  # nothing to compare, e.g. a query made only of stopwords
            return []
        return [article for article in await self._nearest(vector, limit) if article.score > 0]
