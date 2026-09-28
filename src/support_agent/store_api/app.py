"""Mock Store API (FastAPI).

Stands in for the systems a support agent needs in production: the commerce platform
(orders, refunds, addresses), the CRM (customers), the help center (knowledge base) and
the helpdesk (escalations). The agent talks to it over HTTP through `StoreClient`, so
swapping in Shopify/Zendesk/HubSpot means replacing the client, not the agent.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import FastAPI, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse

from support_agent.store_api.repository import (
    ConflictError,
    NotFoundError,
    StoreRepository,
)
from support_agent.store_api.schemas import (
    AddressChangeRequest,
    Customer,
    Escalation,
    EscalationRequest,
    KnowledgeArticle,
    Order,
    Refund,
    RefundRequest,
)


def create_store_app(repository: StoreRepository | None = None) -> FastAPI:
    repo = repository or StoreRepository()
    app = FastAPI(
        title="Mock Store API",
        description="Fictional e-commerce backend used by the support agent's tools.",
        version="1.0.0",
    )
    app.state.repository = repo

    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": str(exc)})

    @app.exception_handler(ConflictError)
    async def _conflict(_: Request, exc: ConflictError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})

    @app.get("/orders/{order_id}", response_model=Order, tags=["orders"])
    async def get_order(order_id: str) -> Order:
        return repo.get_order(order_id)

    @app.post(
        "/orders/{order_id}/refunds",
        response_model=Refund,
        status_code=status.HTTP_201_CREATED,
        tags=["orders"],
    )
    async def create_refund(order_id: str, body: RefundRequest) -> Refund:
        return repo.create_refund(order_id, body.amount, body.reason)

    @app.put("/orders/{order_id}/shipping-address", response_model=Order, tags=["orders"])
    async def update_shipping_address(order_id: str, body: AddressChangeRequest) -> Order:
        return repo.update_shipping_address(order_id, body.shipping_address)

    @app.get("/customers/{email}", response_model=Customer, tags=["customers"])
    async def get_customer(email: str) -> Customer:
        return repo.get_customer(email)

    @app.get("/kb/search", response_model=list[KnowledgeArticle], tags=["knowledge-base"])
    async def search_kb(
        q: Annotated[str, Query(min_length=2, max_length=500)],
        limit: Annotated[int, Query(ge=1, le=10)] = 3,
    ) -> list[KnowledgeArticle]:
        return repo.search_knowledge_base(q, limit=limit)

    @app.post(
        "/escalations",
        response_model=Escalation,
        status_code=status.HTTP_201_CREATED,
        tags=["helpdesk"],
    )
    async def create_escalation(body: EscalationRequest) -> Escalation:
        if not body.reason.strip():
            raise HTTPException(status_code=422, detail="reason must not be empty")
        return repo.create_escalation(body)

    return app
