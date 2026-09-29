"""Application service: submit tickets, run the graph, and resume it after operator review."""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections import defaultdict
from collections.abc import Awaitable, Callable, Sequence
from datetime import datetime
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from pydantic import BaseModel

from support_agent.graph.builder import SupportGraph
from support_agent.logging_config import get_logger
from support_agent.models import (
    ApprovalResponse,
    AuditEvent,
    AuditKind,
    Channel,
    Ticket,
    TicketIn,
    TicketStatus,
    snippet,
    utcnow,
)
from support_agent.persistence import TicketRepository

log = get_logger(__name__)

EventCallback = Callable[[AuditEvent], Awaitable[None] | None]


class TicketNotFoundError(LookupError):
    pass


class InvalidTicketStateError(RuntimeError):
    pass


class AuditEventView(BaseModel):
    seq: int
    at: datetime
    node: str
    kind: AuditKind
    message: str
    data: dict[str, Any]


class RetrievedArticleView(BaseModel):
    id: str
    title: str
    score: float
    snippet: str
    in_reply_context: bool
    cited: bool


class KnowledgeRetrieval(BaseModel):
    """What the knowledge-base search returned and which articles the reply used."""

    query: str
    articles: list[RetrievedArticleView]


class TicketView(BaseModel):
    id: str
    status: TicketStatus
    created_at: datetime
    updated_at: datetime
    channel: Channel
    customer_email: str | None
    subject: str | None
    body: str
    intent: str | None
    verdict: str | None
    resolution: str | None
    analysis: dict[str, Any] | None
    decision: dict[str, Any] | None
    pending_approval: dict[str, Any] | None
    reply: str | None
    error: str | None
    knowledge: KnowledgeRetrieval | None = None
    audit: list[AuditEventView]


def knowledge_from_audit(audit: Sequence[dict[str, Any]]) -> KnowledgeRetrieval | None:
    """Rebuild the retrieval summary from the audit trail (the last search and the drafted reply)."""
    search = next(
        (
            e["data"]["result"]
            for e in reversed(audit)
            if e["data"].get("tool") == "search_knowledge_base" and "articles" in e["data"].get("result", {})
        ),
        None,
    )
    if search is None:
        return None
    reply: dict[str, Any] = next((e["data"] for e in reversed(audit) if e["kind"] == AuditKind.REPLY), {})
    in_context = {a["id"] for a in reply.get("context", {}).get("kb_articles", [])}
    cited = set(reply.get("cited_article_ids", []))
    return KnowledgeRetrieval(
        query=search["query"],
        articles=[
            RetrievedArticleView(
                id=a["id"],
                title=a["title"],
                score=a["score"],
                snippet=snippet(a["content"]),
                in_reply_context=a["id"] in in_context,
                cited=a["id"] in cited,
            )
            for a in search["articles"]
        ],
    )


def new_ticket_id() -> str:
    return f"TCK-{uuid.uuid4().hex[:8].upper()}"


class TicketService:
    def __init__(self, graph: SupportGraph, repository: TicketRepository) -> None:
        self.graph = graph
        self.repository = repository
        self._locks: defaultdict[str, asyncio.Lock] = defaultdict(asyncio.Lock)

    @staticmethod
    def _config(ticket_id: str) -> RunnableConfig:
        # One LangGraph thread per ticket: the checkpointer keys all state by this id.
        return {"configurable": {"thread_id": ticket_id}}

    async def submit(self, ticket_in: TicketIn, on_event: EventCallback | None = None) -> TicketView:
        ticket = Ticket(id=new_ticket_id(), **ticket_in.model_dump())
        await self.repository.create(ticket)
        log.info("ticket.received", ticket_id=ticket.id, channel=ticket.channel.value)
        async with self._locks[ticket.id]:
            return await self._run(ticket.id, {"ticket": ticket}, on_event)

    async def approve(
        self, ticket_id: str, operator: str, note: str | None = None, on_event: EventCallback | None = None
    ) -> TicketView:
        return await self._review(ticket_id, ApprovalResponse(approved=True, operator=operator, note=note), on_event)

    async def reject(
        self, ticket_id: str, operator: str, note: str | None = None, on_event: EventCallback | None = None
    ) -> TicketView:
        return await self._review(ticket_id, ApprovalResponse(approved=False, operator=operator, note=note), on_event)

    async def get(self, ticket_id: str) -> TicketView:
        row = await self.repository.get(ticket_id)
        if row is None:
            raise TicketNotFoundError(ticket_id)
        return TicketView.model_validate({**row, "knowledge": knowledge_from_audit(row["audit"])})

    async def list(self, status: TicketStatus | None = None, limit: int = 50) -> list[dict[str, Any]]:
        return await self.repository.list(status=status, limit=limit)

    async def _review(self, ticket_id: str, response: ApprovalResponse, on_event: EventCallback | None) -> TicketView:
        async with self._locks[ticket_id]:
            current = await self.get(ticket_id)
            if current.status is not TicketStatus.AWAITING_APPROVAL:
                raise InvalidTicketStateError(f"Ticket {ticket_id} is {current.status.value}, not awaiting approval")
            snapshot = await self.graph.aget_state(self._config(ticket_id))
            if not snapshot.interrupts:
                raise InvalidTicketStateError(f"Ticket {ticket_id} has no pending approval in its checkpoint")
            log.info("ticket.reviewed", ticket_id=ticket_id, approved=response.approved, operator=response.operator)
            return await self._run(ticket_id, Command(resume=response.model_dump()), on_event)

    async def _run(self, ticket_id: str, graph_input: Any, on_event: EventCallback | None) -> TicketView:
        config = self._config(ticket_id)
        error: str | None = None
        try:
            async for chunk in self.graph.astream(graph_input, config, stream_mode="updates"):
                for update in chunk.values():
                    if on_event is None or not isinstance(update, dict):
                        continue
                    for event in update.get("audit", []):
                        result = on_event(event)
                        if inspect.isawaitable(result):
                            await result
        except Exception as exc:
            log.exception("ticket.failed", ticket_id=ticket_id)
            error = f"{type(exc).__name__}: {exc}"

        snapshot = await self.graph.aget_state(config)
        values: dict[str, Any] = dict(snapshot.values)
        pending: dict[str, Any] | None = None
        if error is not None:
            status = TicketStatus.FAILED
            failure = AuditEvent(node="runtime", kind=AuditKind.ERROR, message=error)
            values["audit"] = [*values.get("audit", []), failure]
            if on_event is not None:
                result = on_event(failure)
                if inspect.isawaitable(result):
                    await result
        elif snapshot.interrupts:
            status = TicketStatus.AWAITING_APPROVAL
            pending = snapshot.interrupts[0].value
        else:
            # Some checkpointers (e.g. Postgres) store top-level str-enums as plain strings.
            status = TicketStatus(values.get("status", TicketStatus.FAILED))

        await self.repository.save_snapshot(
            ticket_id, status=status, values=values, pending_approval=pending, error=error, at=utcnow()
        )
        log.info("ticket.saved", ticket_id=ticket_id, status=status.value)
        return await self.get(ticket_id)
