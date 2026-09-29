"""Embedding models: OpenAI, or deterministic hashing embeddings that need no model and no network.

Anthropic does not offer an embeddings API, so `EMBEDDINGS_PROVIDER` is independent of
`LLM_PROVIDER`: Claude can classify and draft while OpenAI embeddings power retrieval. Any other
LangChain `Embeddings` implementation (Voyage, Cohere, a local model) plugs in the same way.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from langchain_core.embeddings import Embeddings

from support_agent.config import Settings

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it its me my of on or "
    "our so that the their them there this to was we what when where which will with you your "
    "hi hello thanks please would could".split()
)


def stem(token: str) -> str:
    """Cheap suffix stemming: "shipping"/"shipped"/"ships" -> "ship", "charges"/"charge" -> "charg"."""
    for suffix in ("ing", "ed", "es", "s"):
        if token.endswith(suffix) and len(token) - len(suffix) >= 3:
            token = token[: -len(suffix)]
            break
    if len(token) > 3 and token[-1] == token[-2] and token[-1] not in "aeiouls":
        token = token[:-1]
    if len(token) > 3 and token.endswith("e"):
        token = token[:-1]
    return token


def tokenize(text: str) -> list[str]:
    """Lower-case, stemmed word tokens without stopwords."""
    return [stem(token) for token in _TOKEN_RE.findall(text.lower()) if token not in _STOPWORDS and len(token) >= 2]


def normalize(vector: list[float]) -> list[float]:
    """Scale to unit length so a dot product is the cosine similarity. Zero vectors stay zero."""
    norm = math.sqrt(sum(x * x for x in vector))
    return [x / norm for x in vector] if norm else vector


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embeddings built with the hashing trick.

    Every stemmed token and every pair of adjacent tokens is hashed (BLAKE2b, stable across
    processes) to one of `dimensions` buckets with a +/-1 sign; counts are log-scaled and the
    vector is L2-normalised. Cosine similarity then measures shared vocabulary, which is enough for
    reproducible retrieval in tests, CI, the demo and the eval suite. It understands no synonyms or
    paraphrases: use a real embedding model in production.
    """

    def __init__(self, dimensions: int = 1536) -> None:
        self.dimensions = dimensions

    def _bucket(self, feature: str) -> tuple[int, float]:
        digest = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "big")
        return digest % self.dimensions, 1.0 if (digest >> 63) & 1 else -1.0

    def _embed(self, text: str) -> list[float]:
        tokens = tokenize(text)
        features = Counter(tokens) + Counter(f"{a} {b}" for a, b in pairwise(tokens))
        vector = [0.0] * self.dimensions
        for feature, count in features.items():
            index, sign = self._bucket(feature)
            vector[index] += sign * (1 + math.log(count))
        return normalize(vector)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)

    # Pure CPU and microseconds per text: no need for the base class's thread-pool hop.
    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)


@dataclass(frozen=True)
class EmbeddingModel:
    embeddings: Embeddings
    # Identifies the vector space. Stored next to persisted vectors so a model change re-embeds them.
    name: str
    dimensions: int


def hashing_model(dimensions: int = 1536) -> EmbeddingModel:
    return EmbeddingModel(HashingEmbeddings(dimensions), "hashing-v1", dimensions)


def build_embedding_model(settings: Settings) -> EmbeddingModel:
    dimensions = settings.embeddings_dimensions
    match settings.embeddings_provider:
        case "fake":
            return hashing_model(dimensions)
        case "openai":
            from langchain_openai import OpenAIEmbeddings

            kwargs: dict[str, Any] = {
                "model": settings.openai_embeddings_model,
                "dimensions": dimensions,
                "timeout": settings.llm_timeout_seconds,
                "max_retries": 2,
            }
            if settings.openai_api_key:
                kwargs["api_key"] = settings.openai_api_key
            return EmbeddingModel(OpenAIEmbeddings(**kwargs), f"openai/{settings.openai_embeddings_model}", dimensions)
