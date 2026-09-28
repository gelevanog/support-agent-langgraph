"""In-memory data store behind the mock Store API, seeded from bundled JSON files.

Seed dates are stored as "N days ago" offsets and materialised against a reference date,
so demo scenarios (e.g. "delivered 45 days ago -> outside the refund window") stay
reproducible no matter when the project is run.
"""

from __future__ import annotations

import json
import math
import re
import threading
import uuid
from collections import Counter
from datetime import UTC, date, datetime, timedelta
from decimal import Decimal
from importlib import resources
from typing import Any

from support_agent.store_api.schemas import (
    Customer,
    Escalation,
    EscalationRequest,
    KnowledgeArticle,
    Order,
    OrderStatus,
    Refund,
)

_SEED_PACKAGE = "support_agent.store_api.seed"
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    "a an and are as at be but by can do does for from how i if in is it its me my of on or "
    "our so that the their them there this to was we what when where which will with you your "
    "hi hello thanks please would could".split()
)


class StoreError(Exception):
    """Base class for domain errors raised by the store."""


class NotFoundError(StoreError):
    pass


class ConflictError(StoreError):
    pass


def _tokenize(text: str) -> list[str]:
    tokens = []
    for token in _TOKEN_RE.findall(text.lower()):
        if token in _STOPWORDS or len(token) < 2:
            continue
        # Cheap stemming so "refunds"/"refund", "shipping"/"ship" match.
        for suffix in ("ing", "es", "s"):
            if token.endswith(suffix) and len(token) - len(suffix) >= 3:
                token = token[: -len(suffix)]
                break
        tokens.append(token)
    return tokens


class KnowledgeBaseIndex:
    """Small TF-IDF keyword index. Good enough for a 10-article FAQ; swap for a vector DB at scale."""

    def __init__(self, articles: list[KnowledgeArticle]) -> None:
        self._articles = articles
        self._docs = [Counter(_tokenize(f"{a.title} {a.title} {a.content}")) for a in articles]
        n_docs = len(articles)
        doc_freq: Counter[str] = Counter()
        for doc in self._docs:
            doc_freq.update(doc.keys())
        self._idf = {term: math.log((1 + n_docs) / (1 + df)) + 1 for term, df in doc_freq.items()}
        self._norms = [self._norm(self._weights(doc)) for doc in self._docs]

    def _weights(self, counts: Counter[str]) -> dict[str, float]:
        return {t: (1 + math.log(c)) * self._idf.get(t, 0.0) for t, c in counts.items()}

    @staticmethod
    def _norm(weights: dict[str, float]) -> float:
        return math.sqrt(sum(w * w for w in weights.values())) or 1.0

    def search(self, query: str, limit: int = 3) -> list[KnowledgeArticle]:
        q_weights = self._weights(Counter(_tokenize(query)))
        q_norm = self._norm(q_weights)
        scored: list[tuple[float, KnowledgeArticle]] = []
        for article, doc, d_norm in zip(self._articles, self._docs, self._norms, strict=True):
            d_weights = self._weights(doc)
            dot = sum(w * d_weights.get(t, 0.0) for t, w in q_weights.items())
            score = dot / (q_norm * d_norm)
            if score > 0:
                scored.append((score, article))
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [a.model_copy(update={"score": round(s, 3)}) for s, a in scored[:limit]]


def _load_seed(name: str) -> list[dict[str, Any]]:
    raw = resources.files(_SEED_PACKAGE).joinpath(name).read_text(encoding="utf-8")
    data: list[dict[str, Any]] = json.loads(raw)
    return data


def _order_from_seed(raw: dict[str, Any], today: date) -> Order:
    record = dict(raw)
    for field in ("placed", "shipped", "delivered"):
        days_ago = record.pop(f"{field}_days_ago", None)
        record[f"{field}_on"] = today - timedelta(days=days_ago) if days_ago is not None else None
    return Order.model_validate(record)


class StoreRepository:
    """Thread-safe in-memory store. State lives for the lifetime of the process."""

    def __init__(self, today: date | None = None) -> None:
        self.today = today or datetime.now(UTC).date()
        self._lock = threading.Lock()
        self._orders = {o["id"]: _order_from_seed(o, self.today) for o in _load_seed("orders.json")}
        self._customers = {c["email"].lower(): Customer.model_validate(c) for c in _load_seed("customers.json")}
        self._kb = KnowledgeBaseIndex([KnowledgeArticle.model_validate(a) for a in _load_seed("knowledge_base.json")])
        self.refunds: list[Refund] = []
        self.escalations: list[Escalation] = []

    def get_order(self, order_id: str) -> Order:
        order = self._orders.get(order_id.lstrip("#"))
        if order is None:
            raise NotFoundError(f"Order {order_id} not found")
        return order

    def get_customer(self, email: str) -> Customer:
        customer = self._customers.get(email.strip().lower())
        if customer is None:
            raise NotFoundError(f"Customer {email} not found")
        return customer

    def search_knowledge_base(self, query: str, limit: int = 3) -> list[KnowledgeArticle]:
        return self._kb.search(query, limit=limit)

    def create_refund(self, order_id: str, amount: Decimal, reason: str) -> Refund:
        with self._lock:
            order = self.get_order(order_id)
            if order.status is not OrderStatus.DELIVERED:
                raise ConflictError(f"Order {order_id} is {order.status}, only delivered orders can be refunded")
            if amount > order.refundable_amount:
                raise ConflictError(
                    f"Refund {amount} exceeds refundable amount {order.refundable_amount} for order {order_id}"
                )
            refund = Refund(
                id=f"RF-{uuid.uuid4().hex[:8].upper()}",
                order_id=order.id,
                amount=amount,
                reason=reason,
                created_at=datetime.now(UTC),
            )
            self._orders[order.id] = order.model_copy(update={"refunded_amount": order.refunded_amount + amount})
            self.refunds.append(refund)
            return refund

    def update_shipping_address(self, order_id: str, address: str) -> Order:
        with self._lock:
            order = self.get_order(order_id)
            if order.status is not OrderStatus.PROCESSING:
                raise ConflictError(f"Order {order_id} is {order.status}; address can no longer be changed")
            updated = order.model_copy(update={"shipping_address": address})
            self._orders[order.id] = updated
            return updated

    def create_escalation(self, request: EscalationRequest) -> Escalation:
        with self._lock:
            escalation = Escalation(
                id=f"ESC-{uuid.uuid4().hex[:8].upper()}",
                created_at=datetime.now(UTC),
                **request.model_dump(),
            )
            self.escalations.append(escalation)
            return escalation
