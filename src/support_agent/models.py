"""Agent-side domain models: tickets, LLM analysis, policy decisions and the audit trail."""

from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from support_agent.store_api.schemas import (
    Customer,
    EscalationPriority,
    KnowledgeArticle,
    Order,
)

SNIPPET_CHARS = 160


def utcnow() -> datetime:
    return datetime.now(UTC)


# --- Ticket ---------------------------------------------------------------------------------


class Channel(StrEnum):
    EMAIL = "email"
    CHAT = "chat"
    WEB_FORM = "web_form"


class TicketIn(BaseModel):
    """An incoming support request as it arrives from the helpdesk."""

    body: str = Field(min_length=3, max_length=5000, description="Message text from the customer.")
    customer_email: str | None = Field(
        default=None,
        max_length=254,
        pattern=r"^[^@\s]+@[^@\s]+\.[^@\s]+$",
        description="Verified sender address (from the helpdesk). Omit for anonymous chat.",
    )
    subject: str | None = Field(default=None, max_length=200)
    channel: Channel = Channel.EMAIL


class Ticket(TicketIn):
    id: str
    received_at: datetime = Field(default_factory=utcnow)


class TicketStatus(StrEnum):
    PROCESSING = "processing"
    AWAITING_APPROVAL = "awaiting_approval"
    RESOLVED = "resolved"
    ESCALATED = "escalated"
    FAILED = "failed"


# --- LLM structured output ------------------------------------------------------------------


class Intent(StrEnum):
    ORDER_STATUS = "order_status"
    REFUND_REQUEST = "refund_request"
    PRODUCT_QUESTION = "product_question"
    ADDRESS_CHANGE = "address_change"
    COMPLAINT = "complaint"
    OTHER = "other"


class Urgency(StrEnum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class Sentiment(StrEnum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class TicketAnalysis(BaseModel):
    """Classification and entities extracted from a customer support ticket."""

    intent: Intent = Field(
        description=(
            "Primary intent. refund_request: wants money back or to return an item. "
            "address_change: wants to change the shipping address. order_status: asks where an "
            "order is or when it arrives. product_question: general question about products, "
            "shipping or policies. complaint: expresses dissatisfaction without a concrete request "
            "we can execute. other: anything else."
        )
    )
    urgency: Urgency = Field(
        description="high: threatens chargeback/legal action or is time-critical; low: general questions."
    )
    sentiment: Sentiment = Field(
        description=(
            "Tone of the customer. negative = angry, hostile or threatening. A calm, factual "
            "description of a problem (e.g. 'the item arrived broken') is neutral."
        )
    )
    order_id: str | None = Field(
        description="Order number mentioned in the ticket, digits only without '#'. null if none."
    )
    customer_email: str | None = Field(description="Email address written in the ticket body, if any. null if none.")
    new_shipping_address: str | None = Field(
        description="For address changes: the full new shipping address exactly as written. null otherwise."
    )
    summary: str = Field(description="One-sentence neutral summary of the request for the operator.")


class ReplyDraft(BaseModel):
    """The reply to the customer, written only from the facts in <context>."""

    message: str = Field(
        description=(
            "Reply text: greeting and body, plain text. No signature and no list of article references; "
            "both are appended automatically."
        )
    )
    cited_article_ids: list[str] = Field(
        description=(
            "ids of the help-center articles from <context> (e.g. KB-001) whose information the reply uses. "
            "Empty list if none."
        )
    )


# --- Facts gathered from the store ----------------------------------------------------------


class RetrievedArticle(KnowledgeArticle):
    """A knowledge-base article returned by a search, with its relevance to the query."""

    score: float = Field(description="Cosine similarity between query and article embeddings (higher is closer).")

    @property
    def snippet(self) -> str:
        return snippet(self.content)


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """First `limit` characters of `text`, cut at a word boundary."""
    if len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(",.;:") + "..."


class CaseFacts(BaseModel):
    """Everything the agent learned from store systems. Business rules only look at these."""

    order: Order | None = None
    order_lookup_error: str | None = None
    customer: Customer | None = None
    kb_articles: list[RetrievedArticle] = Field(default_factory=list)
    kb_searched: bool = False


# --- Business rules -------------------------------------------------------------------------


class RuleOutcome(StrEnum):
    """Result of a single rule. Ordered from least to most severe."""

    PASS = "pass"
    NEEDS_APPROVAL = "needs_approval"
    DENY = "deny"
    ESCALATE = "escalate"
    REQUEST_INFO = "request_info"


class RuleCheck(BaseModel):
    rule_id: str
    outcome: RuleOutcome
    detail: str


class ProposedAction(StrEnum):
    REFUND = "refund"
    CHANGE_ADDRESS = "change_address"
    ESCALATE = "escalate"
    NONE = "none"


class Verdict(StrEnum):
    AUTO_APPROVE = "auto_approve"  # execute the action right away
    NEEDS_APPROVAL = "needs_approval"  # pause the graph until an operator decides
    DENY = "deny"  # politely decline, explain why
    ESCALATE = "escalate"  # hand the ticket to a human agent
    INFORM = "inform"  # nothing to execute, answer from facts / knowledge base
    REQUEST_INFO = "request_info"  # ask the customer for missing information


class PolicyDecision(BaseModel):
    action: ProposedAction
    verdict: Verdict
    checks: list[RuleCheck]
    refund_amount: Decimal | None = None
    new_shipping_address: str | None = None
    escalation_priority: EscalationPriority = EscalationPriority.NORMAL

    @property
    def blocking_checks(self) -> list[RuleCheck]:
        return [c for c in self.checks if c.outcome is not RuleOutcome.PASS]

    @property
    def explanation(self) -> str:
        blocking = self.blocking_checks
        return "; ".join(c.detail for c in blocking) if blocking else "All rules passed."


class Resolution(StrEnum):
    """What actually happened to the ticket; drives the customer reply."""

    REFUND_ISSUED = "refund_issued"
    ADDRESS_UPDATED = "address_updated"
    DENIED = "denied"
    REJECTED_AFTER_REVIEW = "rejected_after_review"
    ESCALATED = "escalated"
    INFORMED = "informed"
    NEED_MORE_INFO = "need_more_info"


class ApprovalResponse(BaseModel):
    """What an operator sends to resume a paused ticket."""

    approved: bool
    operator: str = Field(default="operator", min_length=1, max_length=100)
    note: str | None = Field(default=None, max_length=1000)


# --- Audit trail ----------------------------------------------------------------------------


class AuditKind(StrEnum):
    NODE = "node"
    LLM = "llm"
    TOOL = "tool"
    RULE = "rule"
    DECISION = "decision"
    APPROVAL = "approval"
    ACTION = "action"
    REPLY = "reply"
    ERROR = "error"


class AuditEvent(BaseModel):
    at: datetime = Field(default_factory=utcnow)
    node: str
    kind: AuditKind
    message: str
    data: dict[str, Any] = Field(default_factory=dict)
