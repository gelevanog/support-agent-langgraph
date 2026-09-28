"""Unit tests for the deterministic business rules, including boundary values."""

from __future__ import annotations

from datetime import date, timedelta
from decimal import Decimal

import pytest

from support_agent.models import (
    CaseFacts,
    Intent,
    ProposedAction,
    RuleCheck,
    RuleOutcome,
    Sentiment,
    TicketAnalysis,
    Urgency,
    Verdict,
)
from support_agent.rules import policy
from support_agent.rules.policy import PolicyConfig, evaluate
from support_agent.store_api.schemas import (
    Customer,
    CustomerTier,
    EscalationPriority,
    KnowledgeArticle,
    Order,
    OrderItem,
    OrderStatus,
)

TODAY = date(2026, 9, 28)
CONFIG = PolicyConfig()
OWNER = "anna.miller@example.com"


def make_order(
    *,
    status: OrderStatus = OrderStatus.DELIVERED,
    total: str = "64.90",
    delivered_days_ago: int | None = 5,
    refunded: str = "0.00",
    order_id: str = "1042",
) -> Order:
    delivered_on = TODAY - timedelta(days=delivered_days_ago) if delivered_days_ago is not None else None
    return Order(
        id=order_id,
        customer_email=OWNER,
        status=status,
        items=[OrderItem(sku="X", name="Thing", quantity=1, unit_price=Decimal(total))],
        total=Decimal(total),
        placed_on=TODAY - timedelta(days=40),
        shipped_on=TODAY - timedelta(days=10) if status is not OrderStatus.PROCESSING else None,
        delivered_on=delivered_on,
        shipping_address="1 Main St",
        refunded_amount=Decimal(refunded),
    )


def make_customer(tier: CustomerTier = CustomerTier.STANDARD) -> Customer:
    return Customer(
        email=OWNER,
        name="Anna Miller",
        tier=tier,
        customer_since=date(2024, 1, 1),
        orders_count=3,
        lifetime_value=Decimal("200"),
    )


def make_analysis(
    intent: Intent = Intent.REFUND_REQUEST,
    *,
    sentiment: Sentiment = Sentiment.NEUTRAL,
    urgency: Urgency = Urgency.NORMAL,
    order_id: str | None = "1042",
    new_address: str | None = None,
) -> TicketAnalysis:
    return TicketAnalysis(
        intent=intent,
        urgency=urgency,
        sentiment=sentiment,
        order_id=order_id,
        customer_email=None,
        new_shipping_address=new_address,
        summary="test",
    )


def refund_decision(
    order: Order | None = None,
    *,
    customer: Customer | None = None,
    sentiment: Sentiment = Sentiment.NEUTRAL,
    sender: str | None = OWNER,
    config: PolicyConfig = CONFIG,
):
    facts = CaseFacts(order=order or make_order(), customer=customer or make_customer())
    return evaluate(make_analysis(sentiment=sentiment), facts, sender_email=sender, today=TODAY, config=config)


def outcome_of(decision_checks: list[RuleCheck], rule_id: str) -> RuleOutcome:
    return next(c.outcome for c in decision_checks if c.rule_id == rule_id)


# --- refund window ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("days", "expected"),
    [
        (0, RuleOutcome.PASS),
        (29, RuleOutcome.PASS),
        (30, RuleOutcome.PASS),  # exactly 30 days is still inside the window
        (31, RuleOutcome.DENY),
        (90, RuleOutcome.DENY),
    ],
)
def test_refund_window_boundaries(days: int, expected: RuleOutcome) -> None:
    check = policy.check_refund_window(make_order(delivered_days_ago=days), TODAY, CONFIG)
    assert check.outcome is expected


def test_refund_window_is_configurable() -> None:
    config = PolicyConfig(refund_window_days=14)
    assert policy.check_refund_window(make_order(delivered_days_ago=14), TODAY, config).outcome is RuleOutcome.PASS
    assert policy.check_refund_window(make_order(delivered_days_ago=15), TODAY, config).outcome is RuleOutcome.DENY


# --- refund amount ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("amount", "expected"),
    [
        ("0.01", RuleOutcome.PASS),
        ("99.99", RuleOutcome.PASS),
        ("100.00", RuleOutcome.PASS),  # exactly the limit is auto-approved
        ("100.01", RuleOutcome.NEEDS_APPROVAL),
        ("1299.00", RuleOutcome.NEEDS_APPROVAL),
    ],
)
def test_refund_amount_boundaries(amount: str, expected: RuleOutcome) -> None:
    assert policy.check_refund_amount(Decimal(amount), CONFIG).outcome is expected


def test_refund_amount_limit_is_configurable() -> None:
    config = PolicyConfig(auto_approve_refund_limit=Decimal("50"))
    assert policy.check_refund_amount(Decimal("64.90"), config).outcome is RuleOutcome.NEEDS_APPROVAL


# --- other refund rules ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (OrderStatus.DELIVERED, RuleOutcome.PASS),
        (OrderStatus.SHIPPED, RuleOutcome.DENY),
        (OrderStatus.PROCESSING, RuleOutcome.DENY),
        (OrderStatus.CANCELLED, RuleOutcome.DENY),
    ],
)
def test_refund_requires_delivery(status: OrderStatus, expected: RuleOutcome) -> None:
    order = make_order(status=status, delivered_days_ago=5 if status is OrderStatus.DELIVERED else None)
    assert policy.check_refund_delivered(order).outcome is expected


@pytest.mark.parametrize(
    ("refunded", "expected"),
    [("0.00", RuleOutcome.PASS), ("30.00", RuleOutcome.PASS), ("64.90", RuleOutcome.DENY)],
)
def test_not_already_refunded(refunded: str, expected: RuleOutcome) -> None:
    assert policy.check_not_already_refunded(make_order(refunded=refunded)).outcome is expected


def test_partial_refund_only_refunds_the_remaining_balance() -> None:
    decision = refund_decision(make_order(total="150.00", refunded="60.00"))
    assert decision.refund_amount == Decimal("90.00")
    assert decision.verdict is Verdict.AUTO_APPROVE


@pytest.mark.parametrize(
    ("tier", "enabled", "expected"),
    [
        (CustomerTier.VIP, True, RuleOutcome.NEEDS_APPROVAL),
        (CustomerTier.VIP, False, RuleOutcome.PASS),
        (CustomerTier.STANDARD, True, RuleOutcome.PASS),
    ],
)
def test_vip_review(tier: CustomerTier, enabled: bool, expected: RuleOutcome) -> None:
    config = PolicyConfig(vip_refunds_require_approval=enabled)
    assert policy.check_vip_customer(make_customer(tier), config).outcome is expected


def test_vip_check_passes_without_customer_profile() -> None:
    assert policy.check_vip_customer(None, CONFIG).outcome is RuleOutcome.PASS


@pytest.mark.parametrize(
    ("sentiment", "expected"),
    [
        (Sentiment.NEGATIVE, RuleOutcome.NEEDS_APPROVAL),
        (Sentiment.NEUTRAL, RuleOutcome.PASS),
        (Sentiment.POSITIVE, RuleOutcome.PASS),
    ],
)
def test_refund_sentiment_review(sentiment: Sentiment, expected: RuleOutcome) -> None:
    assert policy.check_refund_sentiment(sentiment, CONFIG).outcome is expected


# --- identity --------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("sender", "changes_data", "expected"),
    [
        (OWNER, True, RuleOutcome.PASS),
        ("  Anna.Miller@EXAMPLE.com ", True, RuleOutcome.PASS),  # case/whitespace insensitive
        ("mallory@example.com", True, RuleOutcome.ESCALATE),
        ("mallory@example.com", False, RuleOutcome.ESCALATE),
        (None, True, RuleOutcome.NEEDS_APPROVAL),
        (None, False, RuleOutcome.PASS),
    ],
)
def test_sender_identity(sender: str | None, changes_data: bool, expected: RuleOutcome) -> None:
    check = policy.check_sender_identity(make_order(), sender, CONFIG, changes_data=changes_data)
    assert check.outcome is expected


def test_unverified_sender_rule_can_be_disabled() -> None:
    config = PolicyConfig(unverified_sender_requires_approval=False)
    assert policy.check_sender_identity(make_order(), None, config, changes_data=True).outcome is RuleOutcome.PASS


# --- full refund policy ----------------------------------------------------------------------


def test_refund_auto_approved_when_all_rules_pass() -> None:
    decision = refund_decision()
    assert decision.verdict is Verdict.AUTO_APPROVE
    assert decision.action is ProposedAction.REFUND
    assert decision.refund_amount == Decimal("64.90")
    assert all(c.outcome is RuleOutcome.PASS for c in decision.checks)
    assert decision.explanation == "All rules passed."


def test_refund_high_value_needs_approval() -> None:
    decision = refund_decision(make_order(total="249.00"))
    assert decision.verdict is Verdict.NEEDS_APPROVAL
    assert [c.rule_id for c in decision.blocking_checks] == ["refund.auto_approve_limit"]


def test_refund_vip_needs_approval_even_for_small_amounts() -> None:
    decision = refund_decision(customer=make_customer(CustomerTier.VIP))
    assert decision.verdict is Verdict.NEEDS_APPROVAL
    assert decision.escalation_priority is EscalationPriority.HIGH


def test_refund_negative_sentiment_needs_approval() -> None:
    assert refund_decision(sentiment=Sentiment.NEGATIVE).verdict is Verdict.NEEDS_APPROVAL


def test_deny_beats_needs_approval() -> None:
    # Outside the window AND above the limit: the customer gets a clear "no", not a review.
    decision = refund_decision(make_order(total="500.00", delivered_days_ago=45))
    assert decision.verdict is Verdict.DENY


def test_identity_mismatch_escalates_even_if_refund_would_be_denied() -> None:
    decision = refund_decision(make_order(delivered_days_ago=45), sender="mallory@example.com")
    assert decision.verdict is Verdict.ESCALATE
    assert decision.escalation_priority is EscalationPriority.HIGH


def test_refund_without_order_number_requests_info() -> None:
    decision = evaluate(make_analysis(order_id=None), CaseFacts(), sender_email=OWNER, today=TODAY, config=CONFIG)
    assert decision.verdict is Verdict.REQUEST_INFO
    assert decision.checks[0].detail == "No order number in the ticket."


def test_refund_unknown_order_requests_info() -> None:
    facts = CaseFacts(order_lookup_error="Order 9999 not found")
    decision = evaluate(make_analysis(order_id="9999"), facts, sender_email=OWNER, today=TODAY, config=CONFIG)
    assert decision.verdict is Verdict.REQUEST_INFO
    assert "9999" in decision.checks[0].detail


# --- address change --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("status", "expected"),
    [
        (OrderStatus.PROCESSING, Verdict.AUTO_APPROVE),
        (OrderStatus.SHIPPED, Verdict.DENY),
        (OrderStatus.DELIVERED, Verdict.DENY),
    ],
)
def test_address_change_only_before_shipment(status: OrderStatus, expected: Verdict) -> None:
    order = make_order(status=status, delivered_days_ago=1 if status is OrderStatus.DELIVERED else None)
    analysis = make_analysis(Intent.ADDRESS_CHANGE, new_address="12 Beacon Street, Boston, MA 02108")
    decision = evaluate(analysis, CaseFacts(order=order), sender_email=OWNER, today=TODAY, config=CONFIG)
    assert decision.verdict is expected
    assert decision.action is ProposedAction.CHANGE_ADDRESS


def test_address_change_without_address_requests_info() -> None:
    order = make_order(status=OrderStatus.PROCESSING, delivered_days_ago=None)
    analysis = make_analysis(Intent.ADDRESS_CHANGE, new_address=None)
    decision = evaluate(analysis, CaseFacts(order=order), sender_email=OWNER, today=TODAY, config=CONFIG)
    assert decision.verdict is Verdict.REQUEST_INFO


def test_address_change_from_unverified_sender_needs_approval() -> None:
    order = make_order(status=OrderStatus.PROCESSING, delivered_days_ago=None)
    analysis = make_analysis(Intent.ADDRESS_CHANGE, new_address="12 Beacon Street, Boston")
    decision = evaluate(analysis, CaseFacts(order=order), sender_email=None, today=TODAY, config=CONFIG)
    assert decision.verdict is Verdict.NEEDS_APPROVAL


# --- informational intents -------------------------------------------------------------------


def test_order_status_informs_even_for_unverified_sender() -> None:
    decision = evaluate(
        make_analysis(Intent.ORDER_STATUS),
        CaseFacts(order=make_order()),
        sender_email=None,
        today=TODAY,
        config=CONFIG,
    )
    assert decision.verdict is Verdict.INFORM


@pytest.mark.parametrize(
    ("score", "expected"),
    [(0.15, Verdict.INFORM), (0.40, Verdict.INFORM), (0.149, Verdict.ESCALATE), (0.0, Verdict.ESCALATE)],
)
def test_product_question_needs_a_confident_kb_match(score: float, expected: Verdict) -> None:
    articles = [KnowledgeArticle(id="KB-1", title="Shipping", content="...", score=score)] if score else []
    facts = CaseFacts(kb_articles=articles, kb_searched=True)
    decision = evaluate(
        make_analysis(Intent.PRODUCT_QUESTION, order_id=None), facts, sender_email=None, today=TODAY, config=CONFIG
    )
    assert decision.verdict is expected


@pytest.mark.parametrize(
    ("sentiment", "urgency", "expected"),
    [
        (Sentiment.NEGATIVE, Urgency.NORMAL, Verdict.ESCALATE),
        (Sentiment.NEUTRAL, Urgency.HIGH, Verdict.ESCALATE),
        (Sentiment.NEUTRAL, Urgency.NORMAL, Verdict.INFORM),
    ],
)
def test_complaint_escalation(sentiment: Sentiment, urgency: Urgency, expected: Verdict) -> None:
    analysis = make_analysis(Intent.COMPLAINT, sentiment=sentiment, urgency=urgency)
    decision = evaluate(analysis, CaseFacts(), sender_email=OWNER, today=TODAY, config=CONFIG)
    assert decision.verdict is expected


def test_complaint_escalation_can_be_disabled() -> None:
    analysis = make_analysis(Intent.COMPLAINT, sentiment=Sentiment.NEGATIVE)
    config = PolicyConfig(escalate_negative_complaints=False)
    assert evaluate(analysis, CaseFacts(), sender_email=OWNER, today=TODAY, config=config).verdict is Verdict.INFORM


def test_high_urgency_vip_complaint_gets_high_priority() -> None:
    analysis = make_analysis(Intent.COMPLAINT, sentiment=Sentiment.NEGATIVE, urgency=Urgency.HIGH)
    facts = CaseFacts(customer=make_customer(CustomerTier.VIP))
    decision = evaluate(analysis, facts, sender_email=OWNER, today=TODAY, config=CONFIG)
    assert decision.escalation_priority is EscalationPriority.HIGH


def test_other_intent_is_escalated() -> None:
    decision = evaluate(make_analysis(Intent.OTHER), CaseFacts(), sender_email=OWNER, today=TODAY, config=CONFIG)
    assert decision.verdict is Verdict.ESCALATE
    assert decision.action is ProposedAction.ESCALATE


# --- verdict combination ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("outcomes", "expected"),
    [
        ([], Verdict.AUTO_APPROVE),
        ([RuleOutcome.PASS, RuleOutcome.PASS], Verdict.AUTO_APPROVE),
        ([RuleOutcome.PASS, RuleOutcome.NEEDS_APPROVAL], Verdict.NEEDS_APPROVAL),
        ([RuleOutcome.NEEDS_APPROVAL, RuleOutcome.DENY], Verdict.DENY),
        ([RuleOutcome.DENY, RuleOutcome.ESCALATE], Verdict.ESCALATE),
        ([RuleOutcome.ESCALATE, RuleOutcome.REQUEST_INFO], Verdict.REQUEST_INFO),
    ],
)
def test_combine_takes_most_severe_outcome(outcomes: list[RuleOutcome], expected: Verdict) -> None:
    checks = [RuleCheck(rule_id=f"r{i}", outcome=o, detail="") for i, o in enumerate(outcomes)]
    assert policy.combine(checks, on_pass=Verdict.AUTO_APPROVE) is expected
