"""Operator UI: server-rendered Jinja2 pages, progressively enhanced with htmx.

Works without JavaScript (plain forms + redirects); with htmx, approve/reject swap the card in place.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from support_agent.examples import load_examples
from support_agent.models import Channel, TicketIn, TicketStatus
from support_agent.runtime import Runtime

_HERE = Path(__file__).parent
templates = Jinja2Templates(directory=_HERE / "templates")
router = APIRouter(prefix="/ui", include_in_schema=False)


def _runtime(request: Request) -> Runtime:
    runtime: Runtime = request.app.state.runtime
    return runtime


def _is_htmx(request: Request) -> bool:
    return request.headers.get("HX-Request") == "true"


@router.get("", response_class=HTMLResponse)
async def index(request: Request) -> Response:
    runtime = _runtime(request)
    service = runtime.service
    pending_rows = await service.list(status=TicketStatus.AWAITING_APPROVAL, limit=50)
    pending = [await service.get(row["id"]) for row in pending_rows]
    recent = await service.list(limit=25)
    counts = await service.repository.count_by_status()
    examples = load_examples(runtime.settings.examples_dir)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "pending": pending,
            "recent": recent,
            "counts": counts,
            "examples": examples,
            "example_tickets": [ex.ticket.model_dump(mode="json") for ex in examples],
            "channels": list(Channel),
            "provider": runtime.settings.llm_provider,
        },
    )


@router.get("/tickets/{ticket_id}", response_class=HTMLResponse)
async def ticket_detail(request: Request, ticket_id: str) -> Response:
    ticket = await _runtime(request).service.get(ticket_id)
    return templates.TemplateResponse(request, "ticket.html", {"t": ticket})


@router.post("/tickets")
async def submit_ticket(
    request: Request,
    body: Annotated[str, Form(min_length=3, max_length=5000)],
    customer_email: Annotated[str, Form()] = "",
    subject: Annotated[str, Form()] = "",
    channel: Annotated[Channel, Form()] = Channel.EMAIL,
) -> Response:
    ticket_in = TicketIn(
        body=body,
        customer_email=customer_email.strip() or None,
        subject=subject.strip() or None,
        channel=channel,
    )
    ticket = await _runtime(request).service.submit(ticket_in)
    return RedirectResponse(url=f"/ui/tickets/{ticket.id}", status_code=303)


async def _review(request: Request, ticket_id: str, approved: bool, operator: str, note: str) -> Response:
    service = _runtime(request).service
    operator = operator.strip() or "operator"
    note_value = note.strip() or None
    if approved:
        ticket = await service.approve(ticket_id, operator=operator, note=note_value)
    else:
        ticket = await service.reject(ticket_id, operator=operator, note=note_value)
    if _is_htmx(request):
        context: dict[str, Any] = {"t": ticket}
        return templates.TemplateResponse(request, "partials/review_result.html", context)
    return RedirectResponse(url=f"/ui/tickets/{ticket_id}", status_code=303)


@router.post("/tickets/{ticket_id}/approve")
async def approve(
    request: Request,
    ticket_id: str,
    operator: Annotated[str, Form()] = "operator",
    note: Annotated[str, Form()] = "",
) -> Response:
    return await _review(request, ticket_id, True, operator, note)


@router.post("/tickets/{ticket_id}/reject")
async def reject(
    request: Request,
    ticket_id: str,
    operator: Annotated[str, Form()] = "operator",
    note: Annotated[str, Form()] = "",
) -> Response:
    return await _review(request, ticket_id, False, operator, note)


def mount_ui(app: FastAPI) -> None:
    app.mount("/ui/static", StaticFiles(directory=_HERE / "static"), name="ui-static")
    app.include_router(router)
