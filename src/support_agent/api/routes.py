"""REST API for submitting tickets and reviewing agent decisions."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, Query, Request, status
from pydantic import BaseModel, Field

from support_agent import __version__
from support_agent.models import TicketIn, TicketStatus
from support_agent.runtime import Runtime
from support_agent.service import TicketService, TicketView

router = APIRouter()


def get_runtime(request: Request) -> Runtime:
    runtime: Runtime = request.app.state.runtime
    return runtime


def get_service(runtime: Annotated[Runtime, Depends(get_runtime)]) -> TicketService:
    return runtime.service


ServiceDep = Annotated[TicketService, Depends(get_service)]


class HealthResponse(BaseModel):
    status: str
    version: str
    llm_provider: str
    tickets_by_status: dict[str, int]


class TicketSummary(BaseModel):
    id: str
    status: TicketStatus
    created_at: datetime
    customer_email: str | None
    subject: str | None
    intent: str | None
    verdict: str | None
    resolution: str | None
    summary: str | None


class ReviewRequest(BaseModel):
    operator: str = Field(default="operator", min_length=1, max_length=100)
    note: str | None = Field(default=None, max_length=1000)


def to_summary(row: dict[str, Any]) -> TicketSummary:
    analysis = row.get("analysis") or {}
    return TicketSummary.model_validate({**row, "summary": analysis.get("summary")})


@router.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(runtime: Annotated[Runtime, Depends(get_runtime)]) -> HealthResponse:
    counts = await runtime.service.repository.count_by_status()
    return HealthResponse(
        status="ok",
        version=__version__,
        llm_provider=runtime.settings.llm_provider,
        tickets_by_status=counts,
    )


@router.post("/tickets", response_model=TicketView, status_code=status.HTTP_201_CREATED, tags=["tickets"])
async def submit_ticket(ticket: TicketIn, service: ServiceDep) -> TicketView:
    """Submit a ticket and run the agent until it resolves, escalates or needs approval."""
    return await service.submit(ticket)


@router.get("/tickets", response_model=list[TicketSummary], tags=["tickets"])
async def list_tickets(
    service: ServiceDep,
    status: Annotated[TicketStatus | None, Query(description="Filter, e.g. awaiting_approval")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
) -> list[TicketSummary]:
    return [to_summary(row) for row in await service.list(status=status, limit=limit)]


@router.get("/tickets/{ticket_id}", response_model=TicketView, tags=["tickets"])
async def get_ticket(ticket_id: str, service: ServiceDep) -> TicketView:
    """Full state: classification, rule checks, decision, draft reply and audit trail."""
    return await service.get(ticket_id)


@router.post("/tickets/{ticket_id}/approve", response_model=TicketView, tags=["review"])
async def approve_ticket(
    ticket_id: str, service: ServiceDep, review: Annotated[ReviewRequest | None, Body()] = None
) -> TicketView:
    """Approve the paused action; the graph resumes from its checkpoint and executes it."""
    review = review or ReviewRequest()
    return await service.approve(ticket_id, operator=review.operator, note=review.note)


@router.post("/tickets/{ticket_id}/reject", response_model=TicketView, tags=["review"])
async def reject_ticket(
    ticket_id: str, service: ServiceDep, review: Annotated[ReviewRequest | None, Body()] = None
) -> TicketView:
    """Reject the paused action; the graph resumes and drafts a polite decline."""
    review = review or ReviewRequest()
    return await service.reject(ticket_id, operator=review.operator, note=review.note)
