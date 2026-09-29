"""LangChain tools wrapping the Store API and the knowledge base.

Read-only tools (`get_order`, `get_customer`, `search_knowledge_base`) are bound to the LLM so
it can decide what to look up. Write tools (`create_refund`, `update_shipping_address`,
`escalate_to_human`) are never given to the LLM: graph nodes call them only after the
business rules (and, if required, a human) approved the action.

Every tool returns `(content, artifact)`: compact JSON for the model and the parsed payload as
the artifact, which the graph turns into typed facts for the rules engine.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from langchain_core.tools import BaseTool, tool
from pydantic import BaseModel

from support_agent.knowledge import KnowledgeBase
from support_agent.store_api.schemas import EscalationPriority
from support_agent.tools.client import StoreAPIError, StoreClient

KB_SEARCH_LIMIT = 3

ToolResult = tuple[str, dict[str, Any]]


def _ok(model: BaseModel | Sequence[BaseModel], key: str) -> ToolResult:
    if isinstance(model, BaseModel):
        payload: dict[str, Any] = {key: model.model_dump(mode="json")}
    else:
        payload = {key: [m.model_dump(mode="json") for m in model]}
    return json.dumps(payload, ensure_ascii=False), payload


def _error(exc: StoreAPIError) -> ToolResult:
    payload = {"error": exc.detail, "status_code": exc.status_code}
    return json.dumps(payload), payload


@dataclass(frozen=True)
class StoreTools:
    get_order: BaseTool
    get_customer: BaseTool
    search_knowledge_base: BaseTool
    create_refund: BaseTool
    update_shipping_address: BaseTool
    escalate_to_human: BaseTool

    @property
    def read_only(self) -> list[BaseTool]:
        return [self.get_order, self.get_customer, self.search_knowledge_base]


def build_store_tools(client: StoreClient, knowledge_base: KnowledgeBase) -> StoreTools:
    @tool(response_format="content_and_artifact")
    async def get_order(order_id: str) -> ToolResult:
        """Look up an order by its number (digits only, e.g. "1042").

        Returns status, items, total, delivery dates, shipping address, tracking and refunds.
        """
        try:
            return _ok(await client.get_order(order_id), "order")
        except StoreAPIError as exc:
            return _error(exc)

    @tool(response_format="content_and_artifact")
    async def get_customer(email: str) -> ToolResult:
        """Look up a customer profile (name, tier such as vip, order history) by email address."""
        try:
            return _ok(await client.get_customer(email), "customer")
        except StoreAPIError as exc:
            return _error(exc)

    @tool(response_format="content_and_artifact")
    async def search_knowledge_base(query: str) -> ToolResult:
        """Semantic search over the store's help-center articles (shipping, returns, warranty, payments,
        subscriptions, care, invoices, parts).

        Use a short natural-language query describing the customer's question. Returns the best
        matches with id, title, a snippet and a relevance score between 0 and 1.
        """
        hits = await knowledge_base.search(query, limit=KB_SEARCH_LIMIT)
        # The model sees snippets; the full articles travel in the artifact and become facts.
        content = {"articles": [{"id": h.id, "title": h.title, "snippet": h.snippet, "score": h.score} for h in hits]}
        artifact = {"query": query, "articles": [h.model_dump(mode="json") for h in hits]}
        return json.dumps(content, ensure_ascii=False), artifact

    @tool(response_format="content_and_artifact")
    async def create_refund(order_id: str, amount: str, reason: str) -> ToolResult:
        """Refund `amount` (decimal string) for an order to the original payment method."""
        try:
            return _ok(await client.create_refund(order_id, Decimal(amount), reason), "refund")
        except StoreAPIError as exc:
            return _error(exc)

    @tool(response_format="content_and_artifact")
    async def update_shipping_address(order_id: str, new_address: str) -> ToolResult:
        """Change the shipping address of an order that has not shipped yet."""
        try:
            return _ok(await client.update_shipping_address(order_id, new_address), "order")
        except StoreAPIError as exc:
            return _error(exc)

    @tool(response_format="content_and_artifact")
    async def escalate_to_human(ticket_id: str, reason: str, priority: str = "normal") -> ToolResult:
        """Hand the ticket over to the human support queue with a reason and priority."""
        try:
            escalation = await client.create_escalation(ticket_id, reason, EscalationPriority(priority))
            return _ok(escalation, "escalation")
        except StoreAPIError as exc:
            return _error(exc)

    return StoreTools(
        get_order=get_order,
        get_customer=get_customer,
        search_knowledge_base=search_knowledge_base,
        create_refund=create_refund,
        update_shipping_address=update_shipping_address,
        escalate_to_human=escalate_to_human,
    )
