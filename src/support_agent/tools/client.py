"""HTTP client for the Store API.

This is the adapter boundary: in production you would implement the same methods against
Shopify (orders/refunds), Zendesk/Gorgias (escalations) and your CRM (customers).
"""

from __future__ import annotations

from decimal import Decimal
from types import TracebackType
from typing import Self

import httpx
from fastapi import FastAPI
from pydantic import TypeAdapter

from support_agent.store_api.app import create_store_app
from support_agent.store_api.repository import StoreRepository
from support_agent.store_api.schemas import (
    Customer,
    Escalation,
    EscalationPriority,
    EscalationRequest,
    KnowledgeArticle,
    Order,
    Refund,
)

_ARTICLES = TypeAdapter(list[KnowledgeArticle])


class StoreAPIError(Exception):
    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"Store API error {status_code}: {detail}")
        self.status_code = status_code
        self.detail = detail


class StoreNotFoundError(StoreAPIError):
    pass


class StoreConflictError(StoreAPIError):
    pass


def _raise_for_status(response: httpx.Response) -> None:
    if response.is_success:
        return
    try:
        detail = str(response.json().get("detail", response.text))
    except ValueError:
        detail = response.text
    if response.status_code == 404:
        raise StoreNotFoundError(404, detail)
    if response.status_code == 409:
        raise StoreConflictError(409, detail)
    raise StoreAPIError(response.status_code, detail)


class StoreClient:
    def __init__(self, http: httpx.AsyncClient) -> None:
        self._http = http

    @classmethod
    def in_process(cls, app: FastAPI | None = None) -> Self:
        """Talk to the bundled mock Store API over an in-memory ASGI transport (no network)."""
        transport = httpx.ASGITransport(app=app or create_store_app(StoreRepository()))
        return cls(httpx.AsyncClient(transport=transport, base_url="http://store.internal"))

    @classmethod
    def remote(cls, base_url: str, timeout: float = 10.0) -> Self:
        return cls(httpx.AsyncClient(base_url=base_url, timeout=timeout))

    async def aclose(self) -> None:
        await self._http.aclose()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def get_order(self, order_id: str) -> Order:
        response = await self._http.get(f"/orders/{order_id.lstrip('#')}")
        _raise_for_status(response)
        return Order.model_validate(response.json())

    async def get_customer(self, email: str) -> Customer:
        response = await self._http.get(f"/customers/{email}")
        _raise_for_status(response)
        return Customer.model_validate(response.json())

    async def search_knowledge_base(self, query: str, limit: int = 3) -> list[KnowledgeArticle]:
        response = await self._http.get("/kb/search", params={"q": query, "limit": limit})
        _raise_for_status(response)
        return _ARTICLES.validate_python(response.json())

    async def create_refund(self, order_id: str, amount: Decimal, reason: str) -> Refund:
        response = await self._http.post(f"/orders/{order_id}/refunds", json={"amount": str(amount), "reason": reason})
        _raise_for_status(response)
        return Refund.model_validate(response.json())

    async def update_shipping_address(self, order_id: str, address: str) -> Order:
        response = await self._http.put(f"/orders/{order_id}/shipping-address", json={"shipping_address": address})
        _raise_for_status(response)
        return Order.model_validate(response.json())

    async def create_escalation(self, ticket_id: str, reason: str, priority: EscalationPriority) -> Escalation:
        body = EscalationRequest(ticket_id=ticket_id, reason=reason, priority=priority)
        response = await self._http.post("/escalations", json=body.model_dump(mode="json"))
        _raise_for_status(response)
        return Escalation.model_validate(response.json())
