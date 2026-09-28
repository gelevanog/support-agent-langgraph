"""LangGraph state for one ticket. Persisted by the checkpointer after every step."""

from __future__ import annotations

import operator
from typing import Annotated, Any, TypedDict

from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

from support_agent.models import (
    ApprovalResponse,
    AuditEvent,
    CaseFacts,
    PolicyDecision,
    Resolution,
    Ticket,
    TicketAnalysis,
    TicketStatus,
)


class AgentState(TypedDict, total=False):
    ticket: Ticket
    analysis: TicketAnalysis
    # Conversation of the research (tool-calling) loop only.
    messages: Annotated[list[AnyMessage], add_messages]
    facts: CaseFacts
    decision: PolicyDecision
    approval: ApprovalResponse
    action_result: dict[str, Any]
    action_error: str
    escalation: dict[str, Any]
    resolution: Resolution
    reply: str
    status: TicketStatus
    # Append-only audit trail: every node returns the events it produced.
    audit: Annotated[list[AuditEvent], operator.add]
