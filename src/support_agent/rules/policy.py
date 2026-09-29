"""Deterministic business rules.

The LLM classifies the ticket and gathers facts; *this module* decides what the agent is
allowed to do. Every rule is a small pure function that returns a `RuleCheck`, so each one
is unit-testable, shows up in the audit trail, and can be changed without touching prompts.

Outcome severity (lowest -> highest): pass < needs_approval < deny < escalate < request_info.
The most severe outcome of all checks becomes the verdict.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal

from support_agent.models import (
    CaseFacts,
    Intent,
    PolicyDecision,
    ProposedAction,
    RuleCheck,
    RuleOutcome,
    Sentiment,
    TicketAnalysis,
    Urgency,
    Verdict,
)
from support_agent.store_api.schemas import (
    Customer,
    CustomerTier,
    EscalationPriority,
    Order,
    OrderStatus,
)


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    refund_window_days: int = 30
    auto_approve_refund_limit: Decimal = Decimal("100.00")
    vip_refunds_require_approval: bool = True
    negative_sentiment_requires_approval: bool = True
    unverified_sender_requires_approval: bool = True
    escalate_negative_complaints: bool = True
    kb_min_score: float = 0.15
    address_change_statuses: frozenset[OrderStatus] = frozenset({OrderStatus.PROCESSING})


_SEVERITY: tuple[RuleOutcome, ...] = (
    RuleOutcome.PASS,
    RuleOutcome.NEEDS_APPROVAL,
    RuleOutcome.DENY,
    RuleOutcome.ESCALATE,
    RuleOutcome.REQUEST_INFO,
)
_OUTCOME_TO_VERDICT = {
    RuleOutcome.NEEDS_APPROVAL: Verdict.NEEDS_APPROVAL,
    RuleOutcome.DENY: Verdict.DENY,
    RuleOutcome.ESCALATE: Verdict.ESCALATE,
    RuleOutcome.REQUEST_INFO: Verdict.REQUEST_INFO,
}


def combine(checks: Sequence[RuleCheck], *, on_pass: Verdict) -> Verdict:
    """Collapse individual rule outcomes into a single verdict (most severe wins)."""
    worst = max((c.outcome for c in checks), key=_SEVERITY.index, default=RuleOutcome.PASS)
    return on_pass if worst is RuleOutcome.PASS else _OUTCOME_TO_VERDICT[worst]


# --- Individual rules -----------------------------------------------------------------------


def check_order_found(order_id: str | None, order: Order | None) -> RuleCheck:
    rule = "order.identified"
    if not order_id:
        return RuleCheck(rule_id=rule, outcome=RuleOutcome.REQUEST_INFO, detail="No order number in the ticket.")
    if order is None:
        return RuleCheck(rule_id=rule, outcome=RuleOutcome.REQUEST_INFO, detail=f"Order #{order_id} was not found.")
    return RuleCheck(rule_id=rule, outcome=RuleOutcome.PASS, detail=f"Order #{order.id} found.")


def check_sender_identity(
    order: Order, sender_email: str | None, config: PolicyConfig, *, changes_data: bool
) -> RuleCheck:
    """The verified ticket sender must own the order. Emails typed in the body don't count."""
    rule = "identity.order_owner"
    if sender_email is None:
        if changes_data and config.unverified_sender_requires_approval:
            return RuleCheck(
                rule_id=rule,
                outcome=RuleOutcome.NEEDS_APPROVAL,
                detail="Sender is not verified (no email on the ticket); a human must confirm identity.",
            )
        return RuleCheck(rule_id=rule, outcome=RuleOutcome.PASS, detail="Unverified sender; read-only request.")
    if sender_email.strip().lower() != order.customer_email.lower():
        return RuleCheck(
            rule_id=rule,
            outcome=RuleOutcome.ESCALATE,
            detail=f"Sender {sender_email} is not the owner of order #{order.id}.",
        )
    return RuleCheck(rule_id=rule, outcome=RuleOutcome.PASS, detail="Sender owns the order.")


def check_refund_delivered(order: Order) -> RuleCheck:
    rule = "refund.order_delivered"
    if order.status is OrderStatus.DELIVERED:
        return RuleCheck(rule_id=rule, outcome=RuleOutcome.PASS, detail="Order has been delivered.")
    return RuleCheck(
        rule_id=rule,
        outcome=RuleOutcome.DENY,
        detail=f"Order is {order.status.value}; refunds are available once it has been delivered.",
    )


def check_not_already_refunded(order: Order) -> RuleCheck:
    rule = "refund.not_already_refunded"
    if order.refundable_amount > 0:
        return RuleCheck(
            rule_id=rule, outcome=RuleOutcome.PASS, detail=f"Refundable balance {order.refundable_amount}."
        )
    return RuleCheck(rule_id=rule, outcome=RuleOutcome.DENY, detail="Order has already been fully refunded.")


def check_refund_window(order: Order, today: date, config: PolicyConfig) -> RuleCheck:
    """Refunds are allowed up to and including `refund_window_days` after delivery."""
    rule = "refund.within_window"
    if order.delivered_on is None:
        return RuleCheck(rule_id=rule, outcome=RuleOutcome.PASS, detail="Not delivered yet; window not started.")
    days = (today - order.delivered_on).days
    if days <= config.refund_window_days:
        return RuleCheck(
            rule_id=rule,
            outcome=RuleOutcome.PASS,
            detail=f"Delivered {days} days ago (window: {config.refund_window_days} days).",
        )
    return RuleCheck(
        rule_id=rule,
        outcome=RuleOutcome.DENY,
        detail=f"Delivered {days} days ago, outside the {config.refund_window_days}-day refund window.",
    )


def check_refund_amount(amount: Decimal, config: PolicyConfig) -> RuleCheck:
    """Refunds up to and including the limit are auto-approved; above it a human signs off."""
    rule = "refund.auto_approve_limit"
    limit = config.auto_approve_refund_limit
    if amount <= limit:
        return RuleCheck(
            rule_id=rule, outcome=RuleOutcome.PASS, detail=f"Amount {amount} <= auto-approve limit {limit}."
        )
    return RuleCheck(
        rule_id=rule,
        outcome=RuleOutcome.NEEDS_APPROVAL,
        detail=f"Amount {amount} exceeds auto-approve limit {limit}.",
    )


def check_vip_customer(customer: Customer | None, config: PolicyConfig) -> RuleCheck:
    rule = "refund.vip_review"
    if customer is not None and customer.tier is CustomerTier.VIP and config.vip_refunds_require_approval:
        return RuleCheck(
            rule_id=rule,
            outcome=RuleOutcome.NEEDS_APPROVAL,
            detail="VIP customer: refunds are reviewed by a human.",
        )
    return RuleCheck(rule_id=rule, outcome=RuleOutcome.PASS, detail="Not a VIP account.")


def check_refund_sentiment(sentiment: Sentiment, config: PolicyConfig) -> RuleCheck:
    rule = "refund.sentiment_review"
    if sentiment is Sentiment.NEGATIVE and config.negative_sentiment_requires_approval:
        return RuleCheck(
            rule_id=rule,
            outcome=RuleOutcome.NEEDS_APPROVAL,
            detail="Customer is upset: a human reviews the refund and the tone of the reply.",
        )
    return RuleCheck(rule_id=rule, outcome=RuleOutcome.PASS, detail=f"Sentiment is {sentiment.value}.")


def check_new_address(address: str | None) -> RuleCheck:
    rule = "address.new_address_provided"
    if address and len(address.strip()) >= 5:
        return RuleCheck(rule_id=rule, outcome=RuleOutcome.PASS, detail="New address provided.")
    return RuleCheck(rule_id=rule, outcome=RuleOutcome.REQUEST_INFO, detail="The new shipping address is missing.")


def check_address_changeable(order: Order, config: PolicyConfig) -> RuleCheck:
    rule = "address.before_shipment"
    if order.status in config.address_change_statuses:
        return RuleCheck(rule_id=rule, outcome=RuleOutcome.PASS, detail="Order has not shipped yet.")
    shipped = f" on {order.shipped_on.isoformat()}" if order.shipped_on else ""
    return RuleCheck(
        rule_id=rule,
        outcome=RuleOutcome.DENY,
        detail=f"Order is already {order.status.value}{shipped}; the address can no longer be changed.",
    )


def check_kb_answer(facts: CaseFacts, config: PolicyConfig) -> RuleCheck:
    rule = "kb.answer_found"
    best = max(facts.kb_articles, key=lambda a: a.score, default=None)
    if best is not None and best.score >= config.kb_min_score:
        return RuleCheck(
            rule_id=rule,
            outcome=RuleOutcome.PASS,
            detail=f"Knowledge-base match '{best.title}' [{best.id}] (score {best.score:.2f}).",
        )
    return RuleCheck(
        rule_id=rule,
        outcome=RuleOutcome.ESCALATE,
        detail="No knowledge-base article answers this question; a human should reply.",
    )


def check_complaint(analysis: TicketAnalysis, config: PolicyConfig) -> RuleCheck:
    rule = "complaint.escalation"
    if config.escalate_negative_complaints and (
        analysis.sentiment is Sentiment.NEGATIVE or analysis.urgency is Urgency.HIGH
    ):
        return RuleCheck(
            rule_id=rule,
            outcome=RuleOutcome.ESCALATE,
            detail=f"Complaint with {analysis.sentiment.value} sentiment and {analysis.urgency.value} urgency.",
        )
    return RuleCheck(rule_id=rule, outcome=RuleOutcome.PASS, detail="Mild complaint; acknowledge and log.")


# --- Intent policies ------------------------------------------------------------------------


def _escalation_priority(analysis: TicketAnalysis, facts: CaseFacts, checks: Sequence[RuleCheck]) -> EscalationPriority:
    is_vip = facts.customer is not None and facts.customer.tier is CustomerTier.VIP
    identity_issue = any(c.rule_id == "identity.order_owner" and c.outcome is RuleOutcome.ESCALATE for c in checks)
    if analysis.urgency is Urgency.HIGH or is_vip or identity_issue:
        return EscalationPriority.HIGH
    return EscalationPriority.NORMAL


def evaluate_refund(
    analysis: TicketAnalysis, facts: CaseFacts, sender_email: str | None, today: date, config: PolicyConfig
) -> PolicyDecision:
    found = check_order_found(analysis.order_id, facts.order)
    if facts.order is None:
        return PolicyDecision(action=ProposedAction.REFUND, verdict=Verdict.REQUEST_INFO, checks=[found])
    order = facts.order
    amount = order.refundable_amount
    checks = [
        found,
        check_sender_identity(order, sender_email, config, changes_data=True),
        check_refund_delivered(order),
        check_not_already_refunded(order),
        check_refund_window(order, today, config),
        check_refund_amount(amount, config),
        check_vip_customer(facts.customer, config),
        check_refund_sentiment(analysis.sentiment, config),
    ]
    return PolicyDecision(
        action=ProposedAction.REFUND,
        verdict=combine(checks, on_pass=Verdict.AUTO_APPROVE),
        checks=checks,
        refund_amount=amount,
        escalation_priority=_escalation_priority(analysis, facts, checks),
    )


def evaluate_address_change(
    analysis: TicketAnalysis, facts: CaseFacts, sender_email: str | None, config: PolicyConfig
) -> PolicyDecision:
    found = check_order_found(analysis.order_id, facts.order)
    if facts.order is None:
        return PolicyDecision(action=ProposedAction.CHANGE_ADDRESS, verdict=Verdict.REQUEST_INFO, checks=[found])
    checks = [
        found,
        check_sender_identity(facts.order, sender_email, config, changes_data=True),
        check_new_address(analysis.new_shipping_address),
        check_address_changeable(facts.order, config),
    ]
    return PolicyDecision(
        action=ProposedAction.CHANGE_ADDRESS,
        verdict=combine(checks, on_pass=Verdict.AUTO_APPROVE),
        checks=checks,
        new_shipping_address=analysis.new_shipping_address,
        escalation_priority=_escalation_priority(analysis, facts, checks),
    )


def evaluate_order_status(
    analysis: TicketAnalysis, facts: CaseFacts, sender_email: str | None, config: PolicyConfig
) -> PolicyDecision:
    found = check_order_found(analysis.order_id, facts.order)
    checks = [found]
    if facts.order is not None:
        checks.append(check_sender_identity(facts.order, sender_email, config, changes_data=False))
    return PolicyDecision(
        action=ProposedAction.NONE,
        verdict=combine(checks, on_pass=Verdict.INFORM),
        checks=checks,
        escalation_priority=_escalation_priority(analysis, facts, checks),
    )


def evaluate_product_question(analysis: TicketAnalysis, facts: CaseFacts, config: PolicyConfig) -> PolicyDecision:
    checks = [check_kb_answer(facts, config)]
    verdict = combine(checks, on_pass=Verdict.INFORM)
    return PolicyDecision(
        action=ProposedAction.ESCALATE if verdict is Verdict.ESCALATE else ProposedAction.NONE,
        verdict=verdict,
        checks=checks,
        escalation_priority=_escalation_priority(analysis, facts, checks),
    )


def evaluate_complaint(analysis: TicketAnalysis, facts: CaseFacts, config: PolicyConfig) -> PolicyDecision:
    checks = [check_complaint(analysis, config)]
    verdict = combine(checks, on_pass=Verdict.INFORM)
    return PolicyDecision(
        action=ProposedAction.ESCALATE if verdict is Verdict.ESCALATE else ProposedAction.NONE,
        verdict=verdict,
        checks=checks,
        escalation_priority=_escalation_priority(analysis, facts, checks),
    )


def evaluate_other(analysis: TicketAnalysis, facts: CaseFacts) -> PolicyDecision:
    checks = [
        RuleCheck(
            rule_id="fallback.unsupported_intent",
            outcome=RuleOutcome.ESCALATE,
            detail="Request type is not automated; routing to a human.",
        )
    ]
    return PolicyDecision(
        action=ProposedAction.ESCALATE,
        verdict=Verdict.ESCALATE,
        checks=checks,
        escalation_priority=_escalation_priority(analysis, facts, checks),
    )


def evaluate(
    analysis: TicketAnalysis,
    facts: CaseFacts,
    *,
    sender_email: str | None,
    today: date,
    config: PolicyConfig,
) -> PolicyDecision:
    """Entry point: pick the policy for the ticket's intent and evaluate it."""
    match analysis.intent:
        case Intent.REFUND_REQUEST:
            return evaluate_refund(analysis, facts, sender_email, today, config)
        case Intent.ADDRESS_CHANGE:
            return evaluate_address_change(analysis, facts, sender_email, config)
        case Intent.ORDER_STATUS:
            return evaluate_order_status(analysis, facts, sender_email, config)
        case Intent.PRODUCT_QUESTION:
            return evaluate_product_question(analysis, facts, config)
        case Intent.COMPLAINT:
            return evaluate_complaint(analysis, facts, config)
        case Intent.OTHER:
            return evaluate_other(analysis, facts)
