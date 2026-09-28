"""Persistence: ticket/audit tables (SQLAlchemy) and the LangGraph checkpointer.

One `DATABASE_URL` drives both. SQLite by default; Postgres when the URL starts with
`postgresql` (requires the `postgres` extra).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from support_agent.models import (
    ApprovalResponse,
    AuditEvent,
    AuditKind,
    CaseFacts,
    Channel,
    Intent,
    PolicyDecision,
    ProposedAction,
    Resolution,
    RuleCheck,
    RuleOutcome,
    Sentiment,
    Ticket,
    TicketAnalysis,
    TicketStatus,
    Urgency,
    Verdict,
)
from support_agent.store_api.schemas import (
    Customer,
    CustomerTier,
    EscalationPriority,
    KnowledgeArticle,
    Order,
    OrderItem,
    OrderStatus,
)

# Explicit allowlist of types the checkpointer may deserialise (no arbitrary class loading).
CHECKPOINT_TYPES: tuple[type, ...] = (
    Ticket, Channel, TicketAnalysis, Intent, Urgency, Sentiment, CaseFacts, Order, OrderItem,
    OrderStatus, Customer, CustomerTier, KnowledgeArticle, PolicyDecision, ProposedAction, Verdict,
    RuleCheck, RuleOutcome, EscalationPriority, ApprovalResponse, Resolution, TicketStatus,
    AuditEvent, AuditKind, HumanMessage, AIMessage, SystemMessage, ToolMessage,
)  # fmt: skip

metadata = MetaData()

tickets_table = Table(
    "tickets",
    metadata,
    Column("id", String(32), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("updated_at", DateTime(timezone=True), nullable=False),
    Column("status", String(32), nullable=False, index=True),
    Column("channel", String(16), nullable=False),
    Column("customer_email", String(254)),
    Column("subject", String(200)),
    Column("body", Text, nullable=False),
    Column("intent", String(32)),
    Column("verdict", String(32)),
    Column("resolution", String(32)),
    Column("analysis", JSON),
    Column("decision", JSON),
    Column("pending_approval", JSON),
    Column("reply", Text),
    Column("error", Text),
)

audit_table = Table(
    "audit_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticket_id", String(32), nullable=False, index=True),
    Column("seq", Integer, nullable=False),
    Column("at", DateTime(timezone=True), nullable=False),
    Column("node", String(64), nullable=False),
    Column("kind", String(32), nullable=False),
    Column("message", Text, nullable=False),
    Column("data", JSON, nullable=False),
)


def sqlite_path(database_url: str) -> str | None:
    url = make_url(database_url)
    return (url.database or ":memory:") if url.get_backend_name() == "sqlite" else None


class TicketRepository:
    """Queryable projection of ticket state for the API/UI (the checkpointer is the source of truth
    for resuming graphs; this table is what operators list and filter)."""

    def __init__(self, engine: AsyncEngine) -> None:
        self.engine = engine

    async def create_schema(self) -> None:
        async with self.engine.begin() as conn:
            await conn.run_sync(metadata.create_all)

    async def create(self, ticket: Ticket) -> None:
        async with self.engine.begin() as conn:
            await conn.execute(
                insert(tickets_table).values(
                    id=ticket.id,
                    created_at=ticket.received_at,
                    updated_at=ticket.received_at,
                    status=TicketStatus.PROCESSING.value,
                    channel=ticket.channel.value,
                    customer_email=ticket.customer_email,
                    subject=ticket.subject,
                    body=ticket.body,
                )
            )

    async def save_snapshot(
        self,
        ticket_id: str,
        *,
        status: TicketStatus,
        values: dict[str, Any],
        pending_approval: dict[str, Any] | None,
        error: str | None = None,
        at: datetime,
    ) -> None:
        analysis: TicketAnalysis | None = values.get("analysis")
        decision: PolicyDecision | None = values.get("decision")
        resolution = Resolution(values["resolution"]) if values.get("resolution") else None
        events: Sequence[AuditEvent] = values.get("audit") or []
        async with self.engine.begin() as conn:
            await conn.execute(
                update(tickets_table)
                .where(tickets_table.c.id == ticket_id)
                .values(
                    updated_at=at,
                    status=status.value,
                    intent=analysis.intent.value if analysis else None,
                    verdict=decision.verdict.value if decision else None,
                    resolution=resolution.value if resolution else None,
                    analysis=analysis.model_dump(mode="json") if analysis else None,
                    decision=decision.model_dump(mode="json") if decision else None,
                    pending_approval=pending_approval,
                    reply=values.get("reply"),
                    error=error,
                )
            )
            # The audit trail in graph state is append-only; mirror it in full.
            await conn.execute(delete(audit_table).where(audit_table.c.ticket_id == ticket_id))
            if events:
                await conn.execute(
                    insert(audit_table),
                    [
                        {
                            "ticket_id": ticket_id,
                            "seq": i,
                            "at": e.at,
                            "node": e.node,
                            "kind": e.kind.value,
                            "message": e.message,
                            "data": e.model_dump(mode="json")["data"],
                        }
                        for i, e in enumerate(events)
                    ],
                )

    async def get(self, ticket_id: str) -> dict[str, Any] | None:
        async with self.engine.connect() as conn:
            row = (await conn.execute(select(tickets_table).where(tickets_table.c.id == ticket_id))).mappings().first()
            if row is None:
                return None
            events = (
                await conn.execute(
                    select(audit_table).where(audit_table.c.ticket_id == ticket_id).order_by(audit_table.c.seq)
                )
            ).mappings()
            return {**row, "audit": [dict(e) for e in events]}

    async def list(self, status: TicketStatus | None = None, limit: int = 50) -> list[dict[str, Any]]:
        query = select(tickets_table).order_by(tickets_table.c.created_at.desc()).limit(limit)
        if status is not None:
            query = query.where(tickets_table.c.status == status.value)
        async with self.engine.connect() as conn:
            return [dict(r) for r in (await conn.execute(query)).mappings()]

    async def count_by_status(self) -> dict[str, int]:
        query = select(tickets_table.c.status, func.count()).group_by(tickets_table.c.status)
        async with self.engine.connect() as conn:
            return dict((await conn.execute(query)).all())


def create_engine(database_url: str) -> AsyncEngine:
    path = sqlite_path(database_url)
    if path and path != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    return create_async_engine(database_url)


@asynccontextmanager
async def open_checkpointer(database_url: str) -> AsyncIterator[BaseCheckpointSaver[str]]:
    serde = JsonPlusSerializer(allowed_msgpack_modules=CHECKPOINT_TYPES)
    path = sqlite_path(database_url)
    if path is not None:
        import aiosqlite
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

        async with aiosqlite.connect(path) as conn:
            sqlite_saver = AsyncSqliteSaver(conn, serde=serde)
            await sqlite_saver.setup()
            yield sqlite_saver
        return

    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise RuntimeError("Postgres support requires: uv sync --extra postgres") from exc
    conn_string = make_url(database_url).set(drivername="postgresql").render_as_string(hide_password=False)
    async with AsyncPostgresSaver.from_conn_string(conn_string, serde=serde) as pg_saver:
        await pg_saver.setup()
        yield pg_saver
