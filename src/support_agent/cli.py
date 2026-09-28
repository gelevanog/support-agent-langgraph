"""Command-line interface.

python -m support_agent.cli run "My order #1042 arrived broken, I want my money back"
python -m support_agent.cli example 02 --approve
python -m support_agent.cli demo
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Annotated, TypeVar

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from support_agent.config import get_settings
from support_agent.examples import find_example, load_examples
from support_agent.graph import AgentDeps, build_graph
from support_agent.llm import FakeSupportModel
from support_agent.logging_config import configure_logging
from support_agent.models import AuditEvent, AuditKind, Channel, TicketIn, TicketStatus
from support_agent.runtime import Runtime, create_runtime
from support_agent.service import InvalidTicketStateError, TicketNotFoundError, TicketView
from support_agent.tools import StoreClient, build_store_tools

app = typer.Typer(help="Support Autopilot: resolve support tickets with an AI agent.", no_args_is_help=True)
console = Console(highlight=False)

T = TypeVar("T")

_KIND_STYLE = {
    AuditKind.NODE: "white",
    AuditKind.LLM: "cyan",
    AuditKind.TOOL: "magenta",
    AuditKind.RULE: "yellow",
    AuditKind.DECISION: "bold yellow",
    AuditKind.APPROVAL: "bold red",
    AuditKind.ACTION: "green",
    AuditKind.REPLY: "green",
    AuditKind.ERROR: "bold red",
}
_OUTCOME_STYLE = {
    "pass": "green",
    "needs_approval": "yellow",
    "deny": "red",
    "escalate": "blue",
    "request_info": "blue",
}
_STATUS_STYLE = {
    TicketStatus.RESOLVED: "green",
    TicketStatus.AWAITING_APPROVAL: "yellow",
    TicketStatus.ESCALATED: "blue",
    TicketStatus.FAILED: "red",
    TicketStatus.PROCESSING: "white",
}


def print_event(event: AuditEvent) -> None:
    line = Text()
    line.append(f"{event.node:<15}", style="bold")
    line.append(f"{event.kind.value.upper():<9}", style=_KIND_STYLE[event.kind])
    if event.kind is AuditKind.RULE and "outcome" in event.data:
        outcome = str(event.data["outcome"])
        line.append(f"{outcome:<15}", style=_OUTCOME_STYLE.get(outcome, "white"))
        line.append(f"{event.data['rule_id']:<29}", style="dim")
        line.append(str(event.data["detail"]))
    else:
        line.append(event.message)
    console.print(line, overflow="fold")


def print_ticket_header(ticket: TicketIn, title: str | None = None) -> None:
    sender = ticket.customer_email or "unverified sender"
    body = Text(ticket.body)
    subtitle = f"{ticket.channel.value} from {sender}"
    console.print(Panel(body, title=title or ticket.subject or "New ticket", subtitle=subtitle, expand=False))


def print_outcome(view: TicketView) -> None:
    summary = Table.grid(padding=(0, 2))
    summary.add_row("Ticket", view.id)
    summary.add_row("Status", Text(view.status.value, style=_STATUS_STYLE[view.status]))
    summary.add_row("Intent", view.intent or "-")
    summary.add_row("Verdict", view.verdict or "-")
    summary.add_row("Resolution", view.resolution or "-")
    console.print()
    console.print(summary)
    if view.reply:
        console.print(Panel(view.reply, title="Draft reply", expand=False, border_style="green"))
    if view.status is TicketStatus.AWAITING_APPROVAL:
        reasons = "\n".join(f"- {r}" for r in (view.pending_approval or {}).get("reasons", []))
        console.print(
            Panel(
                f"{reasons}\n\nResume with:\n  python -m support_agent.cli approve {view.id}\n"
                f"  python -m support_agent.cli reject {view.id} --note '...'",
                title="Paused for human approval",
                border_style="yellow",
                expand=False,
            )
        )
    if view.error:
        console.print(Panel(view.error, title="Error", border_style="red"))


def _run(fn: Callable[[Runtime], Awaitable[T]], verbose: bool = False) -> T:
    settings = get_settings()
    configure_logging("INFO" if verbose else "WARNING", settings.log_format)

    async def main() -> T:
        async with create_runtime(settings) as runtime:
            return await fn(runtime)

    try:
        return asyncio.run(main())
    except (TicketNotFoundError, InvalidTicketStateError) as exc:
        message = f"Ticket {exc} not found" if isinstance(exc, TicketNotFoundError) else str(exc)
        console.print(Text(message, style="bold red"))
        raise typer.Exit(code=1) from exc


async def _process(
    runtime: Runtime, ticket: TicketIn, approve: bool, reject: bool, operator: str, note: str | None
) -> TicketView:
    view = await runtime.service.submit(ticket, on_event=print_event)
    if view.status is TicketStatus.AWAITING_APPROVAL and (approve or reject):
        console.print(Text(f"\n--- operator {operator} {'approves' if approve else 'rejects'} ---\n", style="bold"))
        review = runtime.service.approve if approve else runtime.service.reject
        view = await review(view.id, operator, note, on_event=print_event)
    return view


ApproveOpt = Annotated[bool, typer.Option("--approve", help="Auto-approve if the ticket pauses for review.")]
RejectOpt = Annotated[bool, typer.Option("--reject", help="Auto-reject if the ticket pauses for review.")]
OperatorOpt = Annotated[str, typer.Option(help="Operator name recorded in the audit trail.")]
NoteOpt = Annotated[str | None, typer.Option(help="Internal note recorded with the review.")]
VerboseOpt = Annotated[bool, typer.Option("--verbose", "-v", help="Show structured logs.")]


@app.command()
def run(
    text: Annotated[str, typer.Argument(help="The customer's message.")],
    email: Annotated[str | None, typer.Option(help="Verified sender email.")] = None,
    subject: Annotated[str | None, typer.Option()] = None,
    channel: Annotated[Channel, typer.Option()] = Channel.EMAIL,
    approve: ApproveOpt = False,
    reject: RejectOpt = False,
    operator: OperatorOpt = "cli-operator",
    note: NoteOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Run the agent on a ticket and print the step-by-step trace."""
    ticket = TicketIn(body=text, customer_email=email, subject=subject, channel=channel)
    print_ticket_header(ticket)
    view = _run(lambda rt: _process(rt, ticket, approve, reject, operator, note), verbose)
    print_outcome(view)


@app.command()
def example(
    name: Annotated[str, typer.Argument(help="Example number or file name, e.g. 02")],
    approve: ApproveOpt = False,
    reject: RejectOpt = False,
    operator: OperatorOpt = "cli-operator",
    note: NoteOpt = None,
    verbose: VerboseOpt = False,
) -> None:
    """Run one of the demo tickets from examples/."""
    ex = find_example(get_settings().examples_dir, name)
    print_ticket_header(ex.ticket, title=f"{ex.name}: {ex.title}")
    view = _run(lambda rt: _process(rt, ex.ticket, approve, reject, operator, note), verbose)
    print_outcome(view)


@app.command()
def demo() -> None:
    """Run every demo ticket (approving the ones that need review) and print a summary."""
    examples = load_examples(get_settings().examples_dir)
    if not examples:
        raise typer.BadParameter("No examples found; run from the project root or set EXAMPLES_DIR.")

    async def run_all(runtime: Runtime) -> list[tuple[str, TicketView, TicketView]]:
        results = []
        for ex in examples:
            first = await runtime.service.submit(ex.ticket)
            final = first
            if first.status is TicketStatus.AWAITING_APPROVAL:
                final = await runtime.service.approve(first.id, "demo-operator", "Approved in demo run")
            results.append((ex.title, first, final))
        return results

    table = Table(title="Support Autopilot demo")
    table.add_column("Scenario")
    table.add_column("Ticket", no_wrap=True)
    for column in ("Intent", "Verdict", "Review", "Resolution"):
        table.add_column(column)
    for title, first, final in _run(run_all):
        paused = first.status is TicketStatus.AWAITING_APPROVAL
        table.add_row(
            title,
            first.id,
            first.intent or "-",
            first.verdict or "-",
            "approved" if paused else "-",
            Text(final.resolution or final.status.value, style=_STATUS_STYLE[final.status]),
        )
    console.print(table)


@app.command("approve")
def approve_cmd(ticket_id: str, operator: OperatorOpt = "cli-operator", note: NoteOpt = None) -> None:
    """Approve a ticket that is awaiting approval (resumes the graph from its checkpoint)."""
    view = _run(lambda rt: rt.service.approve(ticket_id, operator, note, on_event=print_event))
    print_outcome(view)


@app.command("reject")
def reject_cmd(ticket_id: str, operator: OperatorOpt = "cli-operator", note: NoteOpt = None) -> None:
    """Reject a ticket that is awaiting approval."""
    view = _run(lambda rt: rt.service.reject(ticket_id, operator, note, on_event=print_event))
    print_outcome(view)


@app.command("list")
def list_cmd(
    status: Annotated[TicketStatus | None, typer.Option(help="Filter by status.")] = None,
    limit: int = 20,
) -> None:
    """List recent tickets."""
    rows = _run(lambda rt: rt.service.list(status=status, limit=limit))
    table = Table()
    table.add_column("Ticket", no_wrap=True)
    for column in ("Status", "Intent", "Resolution", "Sender", "Created"):
        table.add_column(column)
    for row in rows:
        row_status = TicketStatus(row["status"])
        table.add_row(
            row["id"],
            Text(row_status.value, style=_STATUS_STYLE[row_status]),
            row["intent"] or "-",
            row["resolution"] or "-",
            row["customer_email"] or "-",
            row["created_at"].strftime("%Y-%m-%d %H:%M"),
        )
    console.print(table)


@app.command()
def show(ticket_id: str) -> None:
    """Show a ticket's audit trail and outcome."""
    view = _run(lambda rt: rt.service.get(ticket_id))
    for event in view.audit:
        print_event(AuditEvent(at=event.at, node=event.node, kind=event.kind, message=event.message, data=event.data))
    print_outcome(view)


@app.command()
def graph() -> None:
    """Print the agent graph as a Mermaid diagram."""

    async def draw() -> str:
        async with StoreClient.in_process() as client:
            compiled = build_graph(AgentDeps(llm=FakeSupportModel(), tools=build_store_tools(client)))
            return compiled.get_graph().draw_mermaid()

    print(asyncio.run(draw()))


if __name__ == "__main__":
    app()
