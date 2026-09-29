"""Graph nodes.

Division of labour:
* LLM nodes (`classify`, `research`, `draft_reply`) understand language and choose lookups.
* Code nodes (`apply_policy`, `execute_action`, `escalate`) decide and act, deterministically.
* `human_approval` pauses the graph with `interrupt()` until an operator responds.
Every node returns audit events, which the state reducer appends to the ticket's trail.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from functools import wraps
from typing import Any, Literal

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolCall, ToolMessage
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.errors import GraphInterrupt
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt

from support_agent.entities import extract_order_id, normalize_order_id
from support_agent.graph.state import AgentState
from support_agent.llm.factory import with_structured_output
from support_agent.llm.prompts import (
    CLASSIFY_SYSTEM,
    DRAFT_SYSTEM,
    RESEARCH_SYSTEM,
    draft_request,
    research_request,
    ticket_block,
)
from support_agent.logging_config import get_logger
from support_agent.models import (
    ApprovalResponse,
    AuditEvent,
    AuditKind,
    CaseFacts,
    Intent,
    PolicyDecision,
    ProposedAction,
    ReplyDraft,
    Resolution,
    RetrievedArticle,
    RuleOutcome,
    Ticket,
    TicketAnalysis,
    TicketStatus,
    Verdict,
)
from support_agent.rules.policy import PolicyConfig, evaluate
from support_agent.store_api.schemas import Customer, Order
from support_agent.tools.definitions import StoreTools

log = get_logger(__name__)

NodeFn = Callable[..., Awaitable[dict[str, Any]]]


def _today_utc() -> date:
    return datetime.now(UTC).date()


@dataclass(frozen=True)
class AgentDeps:
    llm: BaseChatModel
    tools: StoreTools
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    store_name: str = "Brewline Coffee"
    max_research_steps: int = 4
    today: Callable[[], date] = _today_utc


def traced(node: str) -> Callable[[NodeFn], NodeFn]:
    """Log start/finish/duration of a node with the ticket id bound to the log record."""

    def decorator(fn: NodeFn) -> NodeFn:
        @wraps(fn)
        async def wrapper(self: SupportAgentNodes, state: AgentState, **kwargs: Any) -> dict[str, Any]:
            ticket_id = state["ticket"].id
            started = time.perf_counter()
            try:
                result = await fn(self, state, **kwargs)
            except GraphInterrupt:
                log.info("node.interrupted", node=node, ticket_id=ticket_id)
                raise
            except Exception:
                log.exception("node.failed", node=node, ticket_id=ticket_id)
                raise
            log.info(
                "node.completed",
                node=node,
                ticket_id=ticket_id,
                duration_ms=round((time.perf_counter() - started) * 1000, 1),
            )
            return result

        return wrapper

    return decorator


def _call(name: str, args: dict[str, Any]) -> ToolCall:
    return ToolCall(name=name, args=args, id=f"call_{uuid.uuid4().hex[:12]}", type="tool_call")


def _fmt_call(call: ToolCall, *, hide: tuple[str, ...] = ()) -> str:
    """Compact `name(arg=value)` for audit messages; hidden args stay in the event data."""
    args = ", ".join(f"{k}={v!r}" for k, v in call["args"].items() if k not in hide)
    return f"{call['name']}({args})"


def _fmt_result(tool_name: str, artifact: dict[str, Any]) -> str:
    """One-line outcome of a read-only lookup for the audit trail."""
    if tool_name == "search_knowledge_base" and "articles" in artifact:
        hits = ", ".join(f"{a['id']} ({a['score']:.2f})" for a in artifact["articles"])
        return hits or "no matching articles"
    return "ok"


def absorb_tool_result(facts: CaseFacts, tool_name: str, artifact: dict[str, Any]) -> CaseFacts:
    """Turn a read-only tool result into typed facts for the rules engine."""
    match tool_name:
        case "get_order" if "order" in artifact:
            return facts.model_copy(
                update={"order": Order.model_validate(artifact["order"]), "order_lookup_error": None}
            )
        case "get_order":
            return facts.model_copy(update={"order": None, "order_lookup_error": str(artifact.get("error"))})
        case "get_customer" if "customer" in artifact:
            return facts.model_copy(update={"customer": Customer.model_validate(artifact["customer"])})
        case "search_knowledge_base" if "articles" in artifact:
            articles = [RetrievedArticle.model_validate(a) for a in artifact["articles"]]
            return facts.model_copy(update={"kb_articles": articles, "kb_searched": True})
        case _:
            return facts


def determine_resolution(state: AgentState) -> Resolution:
    """What actually happened, derived only from decision, approval and action outcomes."""
    if state.get("escalation"):
        return Resolution.ESCALATED
    decision = state["decision"]
    approval = state.get("approval")
    match decision.verdict:
        case Verdict.AUTO_APPROVE | Verdict.NEEDS_APPROVAL:
            if decision.verdict is Verdict.NEEDS_APPROVAL and not (approval and approval.approved):
                return Resolution.REJECTED_AFTER_REVIEW
            if decision.action is ProposedAction.REFUND:
                return Resolution.REFUND_ISSUED
            return Resolution.ADDRESS_UPDATED
        case Verdict.DENY:
            return Resolution.DENIED
        case Verdict.INFORM:
            return Resolution.INFORMED
        case Verdict.REQUEST_INFO:
            return Resolution.NEED_MORE_INFO
        case Verdict.ESCALATE:
            return Resolution.ESCALATED


def build_reply_context(
    state: AgentState, resolution: Resolution, store_name: str, kb_min_score: float
) -> dict[str, Any]:
    """The facts the reply may use. Nothing outside this dict should reach the customer."""
    ticket, analysis, decision = state["ticket"], state["analysis"], state["decision"]
    facts = state.get("facts") or CaseFacts()
    approval = state.get("approval")
    identity = next((c for c in decision.checks if c.rule_id == "identity.order_owner"), None)
    identity_problem = identity is not None and (
        identity.outcome is RuleOutcome.ESCALATE
        or (identity.outcome is RuleOutcome.NEEDS_APPROVAL and not (approval and approval.approved))
    )
    # Only share order details with the order owner (verified sender, or confirmed by an operator).
    order = facts.order if facts.order and not identity_problem else None
    sender_is_customer = bool(
        facts.customer and ticket.customer_email and facts.customer.email.lower() == ticket.customer_email.lower()
    )
    request_info = next((c for c in decision.checks if c.outcome is RuleOutcome.REQUEST_INFO), None)
    missing_info = None
    if request_info is not None:
        if request_info.rule_id == "address.new_address_provided":
            missing_info = "new_address"
        elif analysis.order_id:
            missing_info = "order_not_found"
        else:
            missing_info = "order_number"
    action_result = state.get("action_result") or {}
    escalation = state.get("escalation")
    return {
        "store_name": store_name,
        "resolution": resolution.value,
        "intent": analysis.intent.value,
        "customer_first_name": facts.customer.name.split()[0] if facts.customer and sender_is_customer else None,
        "order": {
            "id": order.id,
            "status": order.status.value,
            "items": [item.name for item in order.items],
            "total": str(order.total),
            "placed_on": order.placed_on.isoformat(),
            "shipped_on": order.shipped_on.isoformat() if order.shipped_on else None,
            "delivered_on": order.delivered_on.isoformat() if order.delivered_on else None,
            "carrier": order.carrier,
            "tracking_number": order.tracking_number,
        }
        if order
        else None,
        "refund": action_result.get("refund") if resolution is Resolution.REFUND_ISSUED else None,
        "new_shipping_address": decision.new_shipping_address if resolution is Resolution.ADDRESS_UPDATED else None,
        "escalation": {"id": escalation["id"], "priority": escalation["priority"]} if escalation else None,
        "reasons": [c.detail for c in decision.checks if c.outcome is RuleOutcome.DENY]
        if resolution is Resolution.DENIED
        else [],
        "kb_articles": [
            {"id": a.id, "title": a.title, "content": a.content}
            for a in facts.kb_articles[:2]
            if a.score >= kb_min_score
        ],
        "missing_info": missing_info,
        "requested_order_id": analysis.order_id,
    }


def compose_reply(draft: ReplyDraft, context: dict[str, Any]) -> tuple[str, list[str], list[str]]:
    """Final reply text: the model's message, a "See:" line per valid citation, then the signature.

    Citations are checked against the whitelisted context, so the reply can only reference articles
    the rules let through. Returns (reply, cited ids, dropped ids).
    """
    available = {article["id"]: article for article in context["kb_articles"]}
    requested = list(dict.fromkeys(draft.cited_article_ids))
    cited = [article_id for article_id in requested if article_id in available]
    dropped = [article_id for article_id in requested if article_id not in available]
    parts = [draft.message.strip()]
    if cited:
        parts.append("\n".join(f"See: {available[i]['title']} [{i}]" for i in cited))
    parts.append(f"Best regards,\n{context['store_name']} Support")
    return "\n\n".join(parts), cited, dropped


class SupportAgentNodes:
    def __init__(self, deps: AgentDeps) -> None:
        self.deps = deps
        self._lookup_tools: dict[str, BaseTool] = {t.name: t for t in deps.tools.read_only}
        self._tool_node = ToolNode(deps.tools.read_only, handle_tool_errors=True)

    def _system(self, template: str) -> SystemMessage:
        return SystemMessage(content=template.format(store_name=self.deps.store_name))

    # --- 1. Understand ---------------------------------------------------------------------

    @traced("classify")
    async def classify(self, state: AgentState) -> dict[str, Any]:
        ticket = state["ticket"]
        classifier = with_structured_output(self.deps.llm, TicketAnalysis)
        raw = await classifier.ainvoke([self._system(CLASSIFY_SYSTEM), HumanMessage(content=ticket_block(ticket))])
        analysis, notes = self._validate_entities(raw, ticket)
        events = [
            AuditEvent(
                node="classify",
                kind=AuditKind.LLM,
                message=(
                    f"Intent {analysis.intent.value}, urgency {analysis.urgency.value}, "
                    f"sentiment {analysis.sentiment.value}"
                    + (f", order #{analysis.order_id}" if analysis.order_id else "")
                ),
                data=analysis.model_dump(mode="json"),
            )
        ]
        events += [AuditEvent(node="classify", kind=AuditKind.RULE, message=note) for note in notes]
        return {"analysis": analysis, "status": TicketStatus.PROCESSING, "audit": events}

    @staticmethod
    def _validate_entities(analysis: TicketAnalysis, ticket: Ticket) -> tuple[TicketAnalysis, list[str]]:
        """Normalise LLM-extracted identifiers and backfill them with deterministic regexes."""
        notes: list[str] = []
        order_id = normalize_order_id(analysis.order_id)
        regex_order_id = extract_order_id(f"{ticket.subject or ''}\n{ticket.body}")
        if order_id is None and regex_order_id is not None:
            order_id = regex_order_id
            notes.append(f"Order id #{order_id} backfilled by pattern match (missed by the model).")
        email = analysis.customer_email.strip().lower() if analysis.customer_email else None
        return analysis.model_copy(update={"order_id": order_id, "customer_email": email}), notes

    # --- 2. Research (LLM chooses read-only lookups) ------------------------------------------

    @traced("research")
    async def research(self, state: AgentState) -> dict[str, Any]:
        history = list(state.get("messages") or [])
        new_messages: list[Any] = []
        if not history:
            new_messages = [
                self._system(RESEARCH_SYSTEM),
                HumanMessage(content=research_request(state["ticket"], state["analysis"])),
            ]
        model = self.deps.llm.bind_tools(self.deps.tools.read_only)
        response = await model.ainvoke(history + new_messages)
        if response.tool_calls:
            message = "Requested " + ", ".join(_fmt_call(c) for c in response.tool_calls)
        else:
            message = f"Research complete: {response.text.strip() or 'no further lookups needed.'}"
        event = AuditEvent(
            node="research",
            kind=AuditKind.LLM,
            message=message,
            data={"tool_calls": [{"name": c["name"], "args": c["args"]} for c in response.tool_calls]},
        )
        return {"messages": [*new_messages, response], "audit": [event]}

    def route_after_research(self, state: AgentState) -> Literal["lookup_tools", "apply_policy"]:
        messages = state.get("messages") or []
        last = messages[-1] if messages else None
        rounds = sum(1 for m in messages if isinstance(m, AIMessage))
        if isinstance(last, AIMessage) and last.tool_calls and rounds <= self.deps.max_research_steps:
            return "lookup_tools"
        return "apply_policy"

    @traced("lookup_tools")
    async def lookup_tools(self, state: AgentState, config: RunnableConfig) -> dict[str, Any]:
        result = await self._tool_node.ainvoke({"messages": state["messages"]}, config)
        tool_messages: list[ToolMessage] = result["messages"]
        facts = state.get("facts") or CaseFacts()
        events = []
        for msg in tool_messages:
            name = msg.name or ""
            artifact = msg.artifact if isinstance(msg.artifact, dict) else {}
            facts = absorb_tool_result(facts, name, artifact)
            failed = "error" in artifact or msg.status == "error"
            events.append(
                AuditEvent(
                    node="lookup_tools",
                    kind=AuditKind.ERROR if failed else AuditKind.TOOL,
                    message=f"{msg.name} -> "
                    + (f"error: {artifact.get('error', msg.content)}" if failed else _fmt_result(name, artifact)),
                    data={"tool": msg.name, "result": artifact or str(msg.content)},
                )
            )
        return {"messages": tool_messages, "facts": facts, "audit": events}

    # --- 3. Decide (deterministic business rules) ---------------------------------------------

    async def _guard_lookup(self, facts: CaseFacts, name: str, args: dict[str, Any]) -> tuple[CaseFacts, AuditEvent]:
        call = _call(name, args)
        msg: ToolMessage = await self._lookup_tools[name].ainvoke(call)
        artifact = msg.artifact if isinstance(msg.artifact, dict) else {}
        event = AuditEvent(
            node="apply_policy",
            kind=AuditKind.TOOL,
            message=(
                f"Policy guard fetched {_fmt_call(call)} (not provided by research) -> {_fmt_result(name, artifact)}"
            ),
            data={"tool": name, "result": artifact},
        )
        return absorb_tool_result(facts, name, artifact), event

    async def _ensure_required_facts(
        self, analysis: TicketAnalysis, ticket: Ticket, facts: CaseFacts
    ) -> tuple[CaseFacts, list[AuditEvent]]:
        """Rules must never run on missing or mismatched records, whatever the LLM looked up."""
        events: list[AuditEvent] = []
        if analysis.order_id:
            if facts.order is not None and facts.order.id != analysis.order_id:
                facts = facts.model_copy(update={"order": None, "order_lookup_error": None})
            if facts.order is None and facts.order_lookup_error is None:
                facts, event = await self._guard_lookup(facts, "get_order", {"order_id": analysis.order_id})
                events.append(event)
        if analysis.intent in (Intent.REFUND_REQUEST, Intent.COMPLAINT):
            owner = facts.order.customer_email if facts.order else (ticket.customer_email or analysis.customer_email)
            if facts.customer is not None and (owner is None or facts.customer.email.lower() != owner.lower()):
                facts = facts.model_copy(update={"customer": None})
            if facts.customer is None and owner:
                facts, event = await self._guard_lookup(facts, "get_customer", {"email": owner})
                events.append(event)
        if analysis.intent is Intent.PRODUCT_QUESTION and not facts.kb_searched:
            facts, event = await self._guard_lookup(facts, "search_knowledge_base", {"query": ticket.body[:300]})
            events.append(event)
        return facts, events

    @traced("apply_policy")
    async def apply_policy(self, state: AgentState) -> dict[str, Any]:
        ticket, analysis = state["ticket"], state["analysis"]
        facts, events = await self._ensure_required_facts(analysis, ticket, state.get("facts") or CaseFacts())
        decision = evaluate(
            analysis,
            facts,
            sender_email=ticket.customer_email,
            today=self.deps.today(),
            config=self.deps.policy,
        )
        for check in decision.checks:
            events.append(
                AuditEvent(
                    node="apply_policy",
                    kind=AuditKind.RULE,
                    message=f"{check.rule_id}: {check.outcome.value} - {check.detail}",
                    data=check.model_dump(mode="json"),
                )
            )
        amount = f" of {decision.refund_amount}" if decision.refund_amount is not None else ""
        events.append(
            AuditEvent(
                node="apply_policy",
                kind=AuditKind.DECISION,
                message=f"Verdict {decision.verdict.value} for action {decision.action.value}{amount}",
                data=decision.model_dump(mode="json", exclude={"checks"}),
            )
        )
        if decision.verdict is Verdict.NEEDS_APPROVAL:
            events.append(
                AuditEvent(
                    node="apply_policy",
                    kind=AuditKind.APPROVAL,
                    message=f"Approval requested: {decision.explanation}",
                )
            )
        return {"facts": facts, "decision": decision, "audit": events}

    @staticmethod
    def route_by_verdict(
        state: AgentState,
    ) -> Literal["human_approval", "execute_action", "escalate", "draft_reply"]:
        match state["decision"].verdict:
            case Verdict.NEEDS_APPROVAL:
                return "human_approval"
            case Verdict.AUTO_APPROVE:
                return "execute_action"
            case Verdict.ESCALATE:
                return "escalate"
            case _:
                return "draft_reply"

    # --- 4. Human in the loop -------------------------------------------------------------

    @traced("human_approval")
    async def human_approval(self, state: AgentState) -> dict[str, Any]:
        decision: PolicyDecision = state["decision"]
        # Pauses here. The checkpointer persists the state; the graph resumes when the API
        # (or CLI) sends Command(resume=ApprovalResponse(...)).
        response = interrupt(
            {
                "ticket_id": state["ticket"].id,
                "summary": state["analysis"].summary,
                "action": decision.action.value,
                "refund_amount": str(decision.refund_amount) if decision.refund_amount is not None else None,
                "new_shipping_address": decision.new_shipping_address,
                "reasons": [c.detail for c in decision.blocking_checks],
            },
            response_schema=ApprovalResponse,
        )
        verb = "Approved" if response.approved else "Rejected"
        note = f" - {response.note}" if response.note else ""
        event = AuditEvent(
            node="human_approval",
            kind=AuditKind.APPROVAL,
            message=f"{verb} by {response.operator}{note}",
            data=response.model_dump(mode="json"),
        )
        return {"approval": response, "audit": [event]}

    @staticmethod
    def route_after_approval(state: AgentState) -> Literal["execute_action", "draft_reply"]:
        approval = state.get("approval")
        return "execute_action" if approval and approval.approved else "draft_reply"

    # --- 5. Act -----------------------------------------------------------------------------

    @traced("execute_action")
    async def execute_action(self, state: AgentState) -> dict[str, Any]:
        decision, ticket = state["decision"], state["ticket"]
        tools = self.deps.tools
        order_id = state["analysis"].order_id or ""
        if decision.action is ProposedAction.REFUND:
            reason = f"Support ticket {ticket.id}: {state['analysis'].summary}"[:500]
            call = _call(
                "create_refund", {"order_id": order_id, "amount": str(decision.refund_amount), "reason": reason}
            )
            msg: ToolMessage = await tools.create_refund.ainvoke(call)
        elif decision.action is ProposedAction.CHANGE_ADDRESS:
            call = _call(
                "update_shipping_address", {"order_id": order_id, "new_address": decision.new_shipping_address or ""}
            )
            msg = await tools.update_shipping_address.ainvoke(call)
        else:
            raise ValueError(f"Nothing to execute for action {decision.action}")

        artifact = msg.artifact if isinstance(msg.artifact, dict) else {}
        if "error" in artifact:
            event = AuditEvent(
                node="execute_action",
                kind=AuditKind.ERROR,
                message=f"{_fmt_call(call, hide=('reason',))} failed: {artifact['error']}",
                data={"tool": call["name"], "args": call["args"], "result": artifact},
            )
            return {"action_error": str(artifact["error"]), "audit": [event]}
        summary = (
            f"Refund {artifact['refund']['id']} of {artifact['refund']['amount']} created"
            if "refund" in artifact
            else f"Shipping address of order #{order_id} updated"
        )
        event = AuditEvent(
            node="execute_action",
            kind=AuditKind.ACTION,
            message=f"{_fmt_call(call, hide=('reason',))} -> {summary}",
            data={"tool": call["name"], "args": call["args"], "result": artifact},
        )
        return {"action_result": artifact, "audit": [event]}

    @staticmethod
    def route_after_action(state: AgentState) -> Literal["escalate", "draft_reply"]:
        return "escalate" if state.get("action_error") else "draft_reply"

    @traced("escalate")
    async def escalate(self, state: AgentState) -> dict[str, Any]:
        decision, ticket = state["decision"], state["ticket"]
        reason = state.get("action_error") or decision.explanation
        call = _call(
            "escalate_to_human",
            {"ticket_id": ticket.id, "reason": reason[:500], "priority": decision.escalation_priority.value},
        )
        msg: ToolMessage = await self.deps.tools.escalate_to_human.ainvoke(call)
        artifact = msg.artifact if isinstance(msg.artifact, dict) else {}
        if "escalation" not in artifact:
            raise RuntimeError(f"Escalation failed: {artifact.get('error', msg.content)}")
        escalation = artifact["escalation"]
        event = AuditEvent(
            node="escalate",
            kind=AuditKind.ACTION,
            message=f"Escalated to {escalation['queue']} as {escalation['id']} (priority {escalation['priority']})",
            data={"tool": call["name"], "args": call["args"], "result": artifact},
        )
        return {"escalation": escalation, "audit": [event]}

    # --- 6. Respond -------------------------------------------------------------------------

    @traced("draft_reply")
    async def draft_reply(self, state: AgentState) -> dict[str, Any]:
        resolution = determine_resolution(state)
        context = build_reply_context(state, resolution, self.deps.store_name, self.deps.policy.kb_min_score)
        writer = with_structured_output(self.deps.llm, ReplyDraft)
        draft = await writer.ainvoke(
            [self._system(DRAFT_SYSTEM), HumanMessage(content=draft_request(state["ticket"], context))]
        )
        reply, cited, dropped = compose_reply(draft, context)
        status = TicketStatus.ESCALATED if resolution is Resolution.ESCALATED else TicketStatus.RESOLVED
        citations = f", cites {', '.join(cited)}" if cited else ""
        events = [
            AuditEvent(
                node="draft_reply",
                kind=AuditKind.REPLY,
                message=f"Reply drafted ({resolution.value}, {len(reply.split())} words{citations})",
                data={"resolution": resolution.value, "context": context, "cited_article_ids": cited},
            )
        ]
        if dropped:
            events.append(
                AuditEvent(
                    node="draft_reply",
                    kind=AuditKind.RULE,
                    message=f"Dropped citations of articles not in the reply context: {', '.join(dropped)}",
                    data={"dropped_article_ids": dropped},
                )
            )
        return {"reply": reply, "resolution": resolution, "status": status, "audit": events}
