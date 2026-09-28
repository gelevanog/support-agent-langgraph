"""End-to-end graph runs (fake LLM, real tools, real checkpointer) for every demo scenario."""

from __future__ import annotations

import logging
from decimal import Decimal
from typing import Any

import pytest
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from support_agent.config import Settings
from support_agent.llm import FakeSupportModel
from support_agent.models import AuditKind, TicketIn, TicketStatus
from support_agent.runtime import Runtime, create_runtime
from support_agent.service import InvalidTicketStateError, TicketNotFoundError
from tests.conftest import EXAMPLES, example

AUTOMATIC = [e for e in EXAMPLES if e.expected.get("status") != "awaiting_approval"]


@pytest.mark.parametrize("ex", AUTOMATIC, ids=lambda e: e.name)
async def test_example_scenarios(runtime: Runtime, ex: Any) -> None:
    view = await runtime.service.submit(ex.ticket)
    assert view.verdict == ex.expected["verdict"]
    assert view.status.value == ex.expected["status"]
    assert view.resolution == ex.expected["resolution"]
    assert view.reply
    assert view.reply.endswith("Brewline Coffee Support")
    nodes = [e.node for e in view.audit]
    assert nodes[0] == "classify"
    assert nodes[-1] == "draft_reply"
    assert "apply_policy" in nodes


async def test_auto_approved_refund_is_executed(runtime: Runtime) -> None:
    view = await runtime.service.submit(example("01").ticket)
    assert len(runtime.store.refunds) == 1
    refund = runtime.store.refunds[0]
    assert (refund.order_id, refund.amount) == ("1042", Decimal("64.90"))
    assert refund.id in (view.reply or "")
    assert any(e.kind is AuditKind.ACTION and "create_refund" in e.message for e in view.audit)


async def test_denied_refund_explains_window_and_touches_nothing(runtime: Runtime) -> None:
    view = await runtime.service.submit(example("03").ticket)
    assert "outside the 30-day refund window" in (view.reply or "")
    assert runtime.store.refunds == []


async def test_address_change_after_shipment_offers_redirect(runtime: Runtime) -> None:
    view = await runtime.service.submit(example("05").ticket)
    assert "redirect the parcel" in (view.reply or "")
    assert runtime.store.get_order("1046").shipping_address == "310 Pine Rd, Chicago, IL 60601"


async def test_address_change_before_shipment_updates_store(runtime: Runtime) -> None:
    await runtime.service.submit(example("08").ticket)
    assert runtime.store.get_order("1047").shipping_address == "12 Beacon Street, Apt 4, Boston, MA 02108"


async def test_angry_complaint_escalates_with_high_priority(runtime: Runtime) -> None:
    view = await runtime.service.submit(example("07").ticket)
    assert runtime.store.escalations[0].priority == "high"
    assert runtime.store.escalations[0].ticket_id == view.id
    assert "senior member of our support team" in (view.reply or "")


async def test_faq_answer_is_grounded_in_knowledge_base(runtime: Runtime) -> None:
    view = await runtime.service.submit(example("06").ticket)
    assert "Canada" in (view.reply or "")
    assert any(e.node == "lookup_tools" and "search_knowledge_base" in e.message for e in view.audit)


# --- human in the loop -----------------------------------------------------------------------


async def test_needs_approval_pauses_then_approve_resumes(runtime: Runtime) -> None:
    paused = await runtime.service.submit(example("02").ticket)
    assert paused.status is TicketStatus.AWAITING_APPROVAL
    assert paused.pending_approval is not None
    assert paused.pending_approval["refund_amount"] == "249.00"
    assert paused.reply is None
    assert runtime.store.refunds == []

    listed = await runtime.service.list(status=TicketStatus.AWAITING_APPROVAL)
    assert [r["id"] for r in listed] == [paused.id]

    done = await runtime.service.approve(paused.id, operator="maria", note="Checked photos")
    assert done.status is TicketStatus.RESOLVED
    assert done.resolution == "refund_issued"
    assert done.pending_approval is None
    assert runtime.store.refunds[0].amount == Decimal("249.00")
    approval_events = [e for e in done.audit if e.kind is AuditKind.APPROVAL]
    assert approval_events[-1].message == "Approved by maria - Checked photos"
    # The audit trail keeps everything from before the pause.
    assert done.audit[: len(paused.audit)] == paused.audit


async def test_needs_approval_then_reject(runtime: Runtime) -> None:
    paused = await runtime.service.submit(example("02").ticket)
    done = await runtime.service.reject(paused.id, operator="maria", note="Customer abuses returns")
    assert done.status is TicketStatus.RESOLVED
    assert done.resolution == "rejected_after_review"
    assert runtime.store.refunds == []
    assert "not able to approve" in (done.reply or "")
    assert "abuses" not in (done.reply or "")  # internal notes never reach the customer


async def test_cannot_review_twice(runtime: Runtime) -> None:
    paused = await runtime.service.submit(example("02").ticket)
    await runtime.service.approve(paused.id, operator="maria")
    with pytest.raises(InvalidTicketStateError):
        await runtime.service.approve(paused.id, operator="maria")


async def test_cannot_review_ticket_that_did_not_pause(runtime: Runtime) -> None:
    view = await runtime.service.submit(example("04").ticket)
    with pytest.raises(InvalidTicketStateError):
        await runtime.service.reject(view.id, operator="maria")


async def test_unknown_ticket(runtime: Runtime) -> None:
    with pytest.raises(TicketNotFoundError):
        await runtime.service.get("TCK-NOPE")


async def test_resume_after_restart_from_checkpoint(settings: Settings, caplog: pytest.LogCaptureFixture) -> None:
    """The pause survives a process restart: state comes back from the SQLite checkpointer."""
    async with create_runtime(settings) as first:
        paused = await first.service.submit(example("02").ticket)
    caplog.set_level(logging.WARNING)
    async with create_runtime(settings) as second:
        done = await second.service.approve(paused.id, operator="maria")
        assert done.resolution == "refund_issued"
        assert second.store.refunds[0].order_id == "1043"
    # Checkpoint deserialisation uses an explicit type allowlist, so nothing is "unregistered".
    assert "unregistered" not in caplog.text


async def test_vip_refund_needs_approval(runtime: Runtime) -> None:
    ticket = TicketIn(
        customer_email="chloe.nguyen@example.com",
        body="One of the cups from order #1049 has a chip in the rim. Could I get a refund?",
    )
    view = await runtime.service.submit(ticket)
    assert view.status is TicketStatus.AWAITING_APPROVAL
    assert "VIP customer" in " ".join(view.pending_approval["reasons"])  # type: ignore[index]


async def test_unverified_sender_refund_needs_approval_and_greets_after_confirmation(runtime: Runtime) -> None:
    view = await runtime.service.submit(TicketIn(body="My order #1042 arrived broken, I want my money back"))
    assert view.status is TicketStatus.AWAITING_APPROVAL
    done = await runtime.service.approve(view.id, operator="maria", note="Identity confirmed by phone")
    assert done.resolution == "refund_issued"
    assert "order #1042" in (done.reply or "")


async def test_identity_mismatch_escalates_without_leaking_order(runtime: Runtime) -> None:
    ticket = TicketIn(customer_email="mallory@example.com", body="Please refund order #1043, it broke.")
    view = await runtime.service.submit(ticket)
    assert view.status is TicketStatus.ESCALATED
    assert runtime.store.refunds == []
    assert "ProGrind" not in (view.reply or "")
    assert runtime.store.escalations[0].priority == "high"


@pytest.mark.parametrize(
    ("body", "expected_phrase"),
    [
        ("I want a refund for my grinder, it is broken.", "order number"),
        ("Please refund order #9999, it arrived damaged.", "couldn't find order #9999"),
    ],
)
async def test_missing_or_unknown_order_asks_for_info(runtime: Runtime, body: str, expected_phrase: str) -> None:
    view = await runtime.service.submit(TicketIn(customer_email="ben.carter@example.com", body=body))
    assert view.resolution == "need_more_info"
    assert expected_phrase in (view.reply or "")


async def test_store_conflict_during_execution_escalates(runtime: Runtime) -> None:
    paused = await runtime.service.submit(example("02").ticket)
    # Someone refunds the order in the shop admin while the ticket waits for review.
    runtime.store.create_refund("1043", Decimal("249.00"), "manual refund in admin")
    done = await runtime.service.approve(paused.id, operator="maria")
    assert done.status is TicketStatus.ESCALATED
    assert any(e.kind is AuditKind.ERROR and e.node == "execute_action" for e in done.audit)


# --- LLM misbehaviour ------------------------------------------------------------------------


class WrongOrderModel(FakeSupportModel):
    """Research step looks up a different order than the ticket mentions."""

    def _generate(
        self, messages: list[BaseMessage], stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        tool_names = {t["function"]["name"] for t in kwargs.get("tools") or []}
        if tool_names and "TicketAnalysis" not in tool_names:
            already = any(isinstance(m, AIMessage) and m.tool_calls for m in messages)
            call = {"name": "get_order", "args": {"order_id": "1051"}, "id": "c-wrong", "type": "tool_call"}
            message = AIMessage(content="done") if already else AIMessage(content="", tool_calls=[call])
            return ChatResult(generations=[ChatGeneration(message=message)])
        return super()._generate(messages, stop, run_manager, **kwargs)


async def test_policy_guard_ignores_records_the_llm_should_not_have_used(settings: Settings) -> None:
    async with create_runtime(settings, llm=WrongOrderModel()) as rt:
        view = await rt.service.submit(example("02").ticket)
        # Rules ran on order #1043 ($249, needs approval), not the $100 order the model fetched.
        assert view.status is TicketStatus.AWAITING_APPROVAL
        assert view.pending_approval["refund_amount"] == "249.00"  # type: ignore[index]
        assert any("Policy guard fetched get_order" in e.message for e in view.audit)


class FailingModel(FakeSupportModel):
    def _generate(self, *args: Any, **kwargs: Any) -> ChatResult:
        raise RuntimeError("provider unavailable")


async def test_llm_failure_marks_ticket_failed(settings: Settings) -> None:
    async with create_runtime(settings, llm=FailingModel()) as rt:
        view = await rt.service.submit(example("01").ticket)
        assert view.status is TicketStatus.FAILED
        assert "provider unavailable" in (view.error or "")
        assert view.audit[-1].kind is AuditKind.ERROR


async def test_graph_structure(runtime: Runtime) -> None:
    graph = runtime.graph.get_graph()
    assert set(graph.nodes) >= {
        "classify", "research", "lookup_tools", "apply_policy",
        "human_approval", "execute_action", "escalate", "draft_reply",
    }  # fmt: skip
    edges = {(e.source, e.target) for e in graph.edges}
    assert ("apply_policy", "human_approval") in edges
    assert ("human_approval", "execute_action") in edges
    assert ("lookup_tools", "research") in edges
