"""The deterministic fake model and entity extraction."""

from __future__ import annotations

import json

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from support_agent.entities import extract_email, extract_new_address, extract_order_id, normalize_order_id
from support_agent.llm import FakeSupportModel, with_structured_output
from support_agent.llm.fake import analyze_ticket, plan_lookups, render_reply
from support_agent.llm.prompts import research_request, ticket_block
from support_agent.models import Intent, Sentiment, Ticket, TicketAnalysis, Urgency
from support_agent.tools import StoreClient, build_store_tools
from tests.conftest import EXAMPLES


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("My order #1042 arrived broken", "1042"),
        ("order number 1043 is late", "1043"),
        ("Order: 1044", "1044"),
        ("I bought 2 kettles", None),
    ],
)
def test_extract_order_id(text: str, expected: str | None) -> None:
    assert extract_order_id(text) == expected


def test_normalize_order_id() -> None:
    assert normalize_order_id("#1042") == "1042"
    assert normalize_order_id("ORD-1042") == "1042"
    assert normalize_order_id("none") is None
    assert normalize_order_id(None) is None


def test_extract_email() -> None:
    assert extract_email("reach me at Anna.Miller@Example.com please") == "anna.miller@example.com"
    assert extract_email("no email here") is None


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (
            "Could you change the shipping address to 55 Lake Shore Drive, Chicago, IL 60611?",
            "55 Lake Shore Drive, Chicago, IL 60611",
        ),
        (
            "Please update the delivery address for order #1047 to 12 Beacon Street, Boston. Thanks",
            "12 Beacon Street, Boston",
        ),
        ("My new address is 1 Market Plaza, San Francisco", "1 Market Plaza, San Francisco"),
        ("I want to change my address", None),
    ],
)
def test_extract_new_address(text: str, expected: str | None) -> None:
    assert extract_new_address(text) == expected


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.name)
def test_fake_classifier_on_examples(example) -> None:  # type: ignore[no-untyped-def]
    expected_intent = {
        "01": Intent.REFUND_REQUEST,
        "02": Intent.REFUND_REQUEST,
        "03": Intent.REFUND_REQUEST,
        "04": Intent.ORDER_STATUS,
        "05": Intent.ADDRESS_CHANGE,
        "06": Intent.PRODUCT_QUESTION,
        "07": Intent.COMPLAINT,
        "08": Intent.ADDRESS_CHANGE,
    }[example.name[:2]]
    analysis = analyze_ticket(example.ticket.body, example.ticket.subject or "")
    assert analysis.intent is expected_intent


def test_fake_classifier_sentiment_and_urgency() -> None:
    angry = analyze_ticket("This is UNACCEPTABLE, the WORST service. I will dispute the charge!")
    assert angry.sentiment is Sentiment.NEGATIVE
    assert angry.urgency is Urgency.HIGH
    calm = analyze_ticket("My order #1042 arrived broken, I want my money back")
    assert calm.sentiment is Sentiment.NEUTRAL
    assert calm.intent is Intent.REFUND_REQUEST


def test_structured_output_through_standard_interface() -> None:
    ticket = Ticket(id="T-1", body="Where is my order #1045?", customer_email="emma.rossi@example.com")
    classifier = with_structured_output(FakeSupportModel(), TicketAnalysis)
    result = classifier.invoke([SystemMessage(content="classify"), HumanMessage(content=ticket_block(ticket))])
    assert isinstance(result, TicketAnalysis)
    assert result.intent is Intent.ORDER_STATUS
    assert result.order_id == "1045"


async def test_bind_tools_emits_multi_step_plan() -> None:
    ticket = Ticket(id="T-1", body="Refund for order #1042 please", customer_email="anna.miller@example.com")
    analysis = analyze_ticket(ticket.body)
    async with StoreClient.in_process() as client:
        tools = build_store_tools(client)
        model = FakeSupportModel().bind_tools(tools.read_only)
        messages = [SystemMessage(content="research"), HumanMessage(content=research_request(ticket, analysis))]
        first = await model.ainvoke(messages)
        assert [c["name"] for c in first.tool_calls] == ["get_order"]
        order_result = await tools.get_order.ainvoke(first.tool_calls[0])
        second = await model.ainvoke([*messages, first, order_result])
        assert [c["name"] for c in second.tool_calls] == ["get_customer"]
        assert second.tool_calls[0]["args"] == {"email": "anna.miller@example.com"}


def test_plan_stops_when_facts_are_gathered() -> None:
    ticket = Ticket(id="T-1", body="Where is my order #1045?")
    analysis = analyze_ticket(ticket.body)
    human = HumanMessage(content=research_request(ticket, analysis))
    call = {"name": "get_order", "args": {"order_id": "1045"}, "id": "c1", "type": "tool_call"}
    messages = [
        human,
        AIMessage(content="", tool_calls=[call]),
        ToolMessage(content=json.dumps({"order": {"customer_email": "x@y.z"}}), tool_call_id="c1", name="get_order"),
    ]
    assert plan_lookups(messages) == []


def test_render_reply_uses_context_only() -> None:
    reply = render_reply(
        {
            "store_name": "Brewline Coffee",
            "resolution": "refund_issued",
            "intent": "refund_request",
            "customer_first_name": "Anna",
            "order": {"id": "1042"},
            "refund": {"id": "RF-1", "amount": "64.90"},
            "requested_order_id": "1042",
        }
    )
    assert reply.startswith("Hi Anna,")
    assert "$64.90" in reply
    assert "RF-1" in reply
    assert reply.endswith("Brewline Coffee Support")


def test_drafting_without_context_fails_loudly() -> None:
    with pytest.raises(ValueError, match="context"):
        FakeSupportModel().invoke("write something")
