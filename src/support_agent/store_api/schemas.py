"""Data contracts of the (mock) Store API.

These mirror what a real commerce/helpdesk backend exposes (Shopify orders, CRM customers,
help-center articles). The agent only depends on these models, never on the mock internals.
"""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, Field


class OrderStatus(StrEnum):
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"


class CustomerTier(StrEnum):
    STANDARD = "standard"
    VIP = "vip"


class OrderItem(BaseModel):
    sku: str
    name: str
    quantity: int = Field(ge=1)
    unit_price: Decimal


class Order(BaseModel):
    id: str
    customer_email: str
    status: OrderStatus
    items: list[OrderItem]
    total: Decimal
    currency: str = "USD"
    placed_on: date
    shipped_on: date | None = None
    delivered_on: date | None = None
    shipping_address: str
    carrier: str | None = None
    tracking_number: str | None = None
    refunded_amount: Decimal = Decimal("0.00")

    @property
    def refundable_amount(self) -> Decimal:
        return max(self.total - self.refunded_amount, Decimal("0.00"))


class Customer(BaseModel):
    email: str
    name: str
    tier: CustomerTier
    customer_since: date
    orders_count: int
    lifetime_value: Decimal


class KnowledgeArticle(BaseModel):
    """A help-center article (the source of truth the agent's search index is built from)."""

    id: str
    title: str
    content: str


class RefundRequest(BaseModel):
    amount: Decimal = Field(gt=0)
    reason: str = Field(min_length=1, max_length=500)


class Refund(BaseModel):
    id: str
    order_id: str
    amount: Decimal
    reason: str
    created_at: datetime


class AddressChangeRequest(BaseModel):
    shipping_address: str = Field(min_length=5, max_length=300)


class EscalationPriority(StrEnum):
    NORMAL = "normal"
    HIGH = "high"


class EscalationRequest(BaseModel):
    ticket_id: str
    reason: str
    priority: EscalationPriority = EscalationPriority.NORMAL
    queue: str = "tier-2"


class Escalation(BaseModel):
    id: str
    ticket_id: str
    reason: str
    priority: EscalationPriority
    queue: str
    created_at: datetime
