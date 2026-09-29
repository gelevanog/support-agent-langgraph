"""Deterministic offline chat model.

`FakeSupportModel` is a real LangChain `BaseChatModel`: it supports `bind_tools` (emits tool
calls) and therefore `with_structured_output`, and it reads the very same prompts the real
models get. Classification uses keyword rules, research follows a fixed lookup plan and replies
are rendered from templates (citing the help-center articles whose text they use). This lets
the full graph, the test-suite, the demo and the Docker image run with zero API keys and fully
reproducible output.
"""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Sequence
from typing import Any, ClassVar

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models import BaseChatModel, LangSmithParams, LanguageModelInput
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolCall, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool

from support_agent.entities import extract_email, extract_new_address, extract_order_id
from support_agent.models import Intent, ReplyDraft, Resolution, Sentiment, TicketAnalysis, Urgency

_ADDRESS_WORDS = ("shipping address", "delivery address", "change my address", "new address", "wrong address")
_REFUND_WORDS = ("refund", "money back", "reimburse", "return it", "return the", "send it back", "chargeback")
_STATUS_WORDS = (
    "where is",
    "where's",
    "status",
    "tracking",
    "track my",
    "not arrived",
    "hasn't arrived",
    "has not arrived",
    "not received",
    "when will",
    "still waiting",
)
_QUESTION_WORDS = ("do you", "does ", "can i", "how long", "how do", "is it possible", "what is", "warranty", "ship to")
_NEGATIVE_WORDS = (
    "unacceptable",
    "worst",
    "terrible",
    "awful",
    "ridiculous",
    "furious",
    "angry",
    "disgusting",
    "never again",
    "scam",
    "horrible",
    "fed up",
    "incompetent",
    "disappointed",
)
_POSITIVE_WORDS = ("thank", "great", "love", "awesome", "appreciate")
_HIGH_URGENCY_WORDS = (
    "urgent",
    "asap",
    "immediately",
    "today",
    "right now",
    "dispute",
    "chargeback",
    "lawyer",
    "legal",
)
_CAPS_WORD = re.compile(r"\b[A-Z]{3,}\b")


def extract_tag(text: str, tag: str) -> str | None:
    """Content of the first <tag ...>...</tag> block in a prompt."""
    match = re.search(rf"<{tag}[^>]*>\n?(.*?)\n?</{tag}>", text, re.DOTALL)
    return match.group(1) if match else None


def _ticket_parts(text: str) -> tuple[str | None, str, str]:
    """Split the <ticket> block into (verified sender email, subject, body)."""
    block = extract_tag(text, "ticket") or text
    header, _, body = block.partition("\n\n")
    sender_match = re.search(r"^From: (\S+@\S+) \(verified\)", header, re.MULTILINE)
    subject_match = re.search(r"^Subject: (.*)$", header, re.MULTILINE)
    subject = subject_match.group(1) if subject_match and subject_match.group(1) != "(no subject)" else ""
    return (sender_match.group(1) if sender_match else None), subject, body.strip()


def _hits(text: str, words: Sequence[str]) -> int:
    return sum(1 for w in words if w in text)


def analyze_ticket(body: str, subject: str = "") -> TicketAnalysis:
    """Keyword-based ticket classification (the offline stand-in for an LLM)."""
    text = f"{subject}\n{body}".strip()
    lower = text.lower()
    negative = _hits(lower, _NEGATIVE_WORDS)
    shouting = len(_CAPS_WORD.findall(text)) >= 2 or "!!" in text
    order_id = extract_order_id(text)

    if _hits(lower, _ADDRESS_WORDS):
        intent = Intent.ADDRESS_CHANGE
    elif _hits(lower, _REFUND_WORDS):
        intent = Intent.REFUND_REQUEST
    elif negative >= 2 or "complain" in lower:
        intent = Intent.COMPLAINT
    elif _hits(lower, _STATUS_WORDS):
        intent = Intent.ORDER_STATUS
    elif _hits(lower, _QUESTION_WORDS) or "?" in text:
        intent = Intent.PRODUCT_QUESTION
    elif negative:
        intent = Intent.COMPLAINT
    else:
        intent = Intent.OTHER

    if negative or shouting:
        sentiment = Sentiment.NEGATIVE
    elif _hits(lower, _POSITIVE_WORDS):
        sentiment = Sentiment.POSITIVE
    else:
        sentiment = Sentiment.NEUTRAL

    if _hits(lower, _HIGH_URGENCY_WORDS):
        urgency = Urgency.HIGH
    elif intent is Intent.PRODUCT_QUESTION:
        urgency = Urgency.LOW
    else:
        urgency = Urgency.NORMAL

    first_sentence = re.split(r"(?<=[.!?])\s", " ".join(body.split()), maxsplit=1)[0]
    label = intent.value.replace("_", " ").capitalize()
    order_ref = f" for order #{order_id}" if order_id else ""
    return TicketAnalysis(
        intent=intent,
        urgency=urgency,
        sentiment=sentiment,
        order_id=order_id,
        customer_email=extract_email(text),
        new_shipping_address=extract_new_address(text) if intent is Intent.ADDRESS_CHANGE else None,
        summary=f"{label}{order_ref}: {first_sentence[:140]}",
    )


def make_tool_call(name: str, args: dict[str, Any]) -> ToolCall:
    return ToolCall(name=name, args=args, id=f"call_{uuid.uuid4().hex[:12]}", type="tool_call")


def plan_lookups(messages: Sequence[BaseMessage]) -> list[ToolCall]:
    """Decide the next read-only lookups from the conversation so far (fixed, sensible plan)."""
    first_human = next((m for m in messages if isinstance(m, HumanMessage)), None)
    if first_human is None:
        return []
    text = str(first_human.content)
    analysis_json = extract_tag(text, "analysis")
    sender, subject, body = _ticket_parts(text)
    analysis = TicketAnalysis.model_validate_json(analysis_json) if analysis_json else analyze_ticket(body, subject)

    results: dict[str, dict[str, Any]] = {}
    for message in messages:
        if isinstance(message, ToolMessage) and message.name:
            results[message.name] = json.loads(str(message.content))

    calls: list[ToolCall] = []
    wants_customer = analysis.intent in (Intent.REFUND_REQUEST, Intent.COMPLAINT)
    if analysis.order_id and "get_order" not in results:
        calls.append(make_tool_call("get_order", {"order_id": analysis.order_id}))
    if analysis.intent is Intent.PRODUCT_QUESTION and "search_knowledge_base" not in results:
        calls.append(make_tool_call("search_knowledge_base", {"query": body[:300]}))
    if wants_customer and "get_customer" not in results:
        order = results.get("get_order", {}).get("order")
        email = order["customer_email"] if order else (None if analysis.order_id else sender)
        if email:
            calls.append(make_tool_call("get_customer", {"email": email}))
    return calls


# --- Reply templates -------------------------------------------------------------------------


def _items(order: dict[str, Any]) -> str:
    return ", ".join(order.get("items", [])) or "your items"


def _order_ref(ctx: dict[str, Any]) -> str:
    order = ctx.get("order") or {}
    order_id = order.get("id") or ctx.get("requested_order_id")
    return f"order #{order_id}" if order_id else "your order"


def _reply_refund_issued(ctx: dict[str, Any]) -> str:
    refund = ctx["refund"]
    return (
        f"Thank you for letting us know, and I'm sorry {_order_ref(ctx)} didn't work out. "
        f"I've issued a full refund of ${refund['amount']} to your original payment method "
        f"(reference {refund['id']}). There is no need to send anything back."
    )


def _reply_address_updated(ctx: dict[str, Any]) -> str:
    return (
        f"Done! The shipping address for {_order_ref(ctx)} has been updated to:\n"
        f"{ctx['new_shipping_address']}\n\nYour order hasn't shipped yet, so it will go straight there."
    )


def _reply_denied(ctx: dict[str, Any]) -> str:
    order = ctx.get("order")
    reasons = " ".join(ctx.get("reasons", []))
    ref = f" for order #{order['id']}" if order else ""
    what = "change the shipping address" if ctx["intent"] == Intent.ADDRESS_CHANGE else "process a refund"
    text = f"Thanks for reaching out. Unfortunately I can't {what}{ref}. {reasons}"
    if ctx["intent"] == Intent.ADDRESS_CHANGE and order and order.get("tracking_number"):
        text += (
            f"\n\nGood news: you can ask {order['carrier']} to redirect the parcel using tracking "
            f"number {order['tracking_number']}."
        )
    for article in ctx.get("kb_articles", [])[:1]:
        text += f"\n\n{article['content']}"
    return text


def _reply_rejected(ctx: dict[str, Any]) -> str:
    order = ctx.get("order")
    ref = f" for order #{order['id']}" if order else ""
    return (
        f"Thank you for your patience. Our team has reviewed your request{ref} and unfortunately "
        "we are not able to approve it. If you reply to this email, a team member will be happy to "
        "go through the options with you."
    )


def _reply_escalated(ctx: dict[str, Any]) -> str:
    order = ctx.get("order")
    escalation = ctx.get("escalation") or {}
    apology = f"I'm sorry about the trouble with order #{order['id']}. " if order else ""
    speed = "as a priority" if escalation.get("priority") == "high" else "shortly"
    reference = f" (reference {escalation['id']})" if escalation.get("id") else ""
    return (
        f"{apology}I've passed your message to a senior member of our support team{reference}. "
        f"They have the full history and will get back to you {speed}."
    )


def _reply_informed(ctx: dict[str, Any]) -> str:
    order = ctx.get("order")
    if ctx["intent"] == Intent.ORDER_STATUS and order:
        if order["status"] == "delivered":
            return f"Order #{order['id']} ({_items(order)}) was delivered on {order['delivered_on']}."
        if order["status"] == "shipped":
            return (
                f"Good news: order #{order['id']} ({_items(order)}) shipped on {order['shipped_on']} with "
                f"{order['carrier']}. Your tracking number is {order['tracking_number']}; tracking can "
                "take up to 24 hours to show the latest scan."
            )
        return f"Order #{order['id']} ({_items(order)}) is {order['status']} and hasn't shipped yet."
    articles = ctx.get("kb_articles", [])
    if ctx["intent"] == Intent.PRODUCT_QUESTION and articles:
        return f"Great question! {articles[0]['content']}"
    return "Thank you for the feedback. I've shared it with the team so we can do better."


def _reply_need_more_info(ctx: dict[str, Any]) -> str:
    missing = ctx.get("missing_info")
    if missing == "order_not_found":
        return (
            f"I couldn't find order #{ctx.get('requested_order_id')} in our system. Could you double-check "
            "the order number? You'll find it in your confirmation email."
        )
    if missing == "new_address":
        return "Happy to help! Could you reply with the full new shipping address, including the ZIP code?"
    return "Happy to help! Could you reply with your order number? You'll find it in your order confirmation email."


_TEMPLATES: dict[Resolution, Callable[[dict[str, Any]], str]] = {
    Resolution.REFUND_ISSUED: _reply_refund_issued,
    Resolution.ADDRESS_UPDATED: _reply_address_updated,
    Resolution.DENIED: _reply_denied,
    Resolution.REJECTED_AFTER_REVIEW: _reply_rejected,
    Resolution.ESCALATED: _reply_escalated,
    Resolution.INFORMED: _reply_informed,
    Resolution.NEED_MORE_INFO: _reply_need_more_info,
}


def render_reply(context: dict[str, Any]) -> ReplyDraft:
    name = context.get("customer_first_name")
    greeting = f"Hi {name}," if name else "Hi there,"
    body = _TEMPLATES[Resolution(context["resolution"])](context)
    cited = [a["id"] for a in context.get("kb_articles", []) if a["content"] in body]
    return ReplyDraft(message=f"{greeting}\n\n{body}", cited_article_ids=cited)


class OfflineChatModel(BaseChatModel):
    """Base for deterministic chat models: tool binding (and therefore structured output) like the real ones."""

    model_id: ClassVar[str] = "fake"

    @property
    def _llm_type(self) -> str:
        return self.model_id

    def _get_ls_params(self, stop: list[str] | None = None, **kwargs: Any) -> LangSmithParams:
        return LangSmithParams(ls_provider="fake", ls_model_name=self.model_id, ls_model_type="chat")

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        formatted = [convert_to_openai_tool(t) for t in tools]
        return self.bind(tools=formatted, tool_choice=tool_choice, **kwargs)

    @staticmethod
    def bound_tool_names(kwargs: dict[str, Any]) -> set[str]:
        return {t["function"]["name"] for t in kwargs.get("tools") or []}

    @staticmethod
    def last_human_text(messages: Sequence[BaseMessage]) -> str:
        last_human = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)
        return str(last_human.content) if last_human else ""


class FakeSupportModel(OfflineChatModel):
    """Offline, deterministic chat model that speaks the same interface as ChatOpenAI/ChatAnthropic."""

    model_id: ClassVar[str] = "fake-support"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        tool_names = self.bound_tool_names(kwargs)
        prompt = self.last_human_text(messages)

        if TicketAnalysis.__name__ in tool_names:
            _, subject, body = _ticket_parts(prompt)
            analysis = analyze_ticket(body, subject)
            call = make_tool_call(TicketAnalysis.__name__, analysis.model_dump(mode="json"))
            message = AIMessage(content="", tool_calls=[call])
        elif ReplyDraft.__name__ in tool_names:
            context_json = extract_tag(prompt, "context")
            if context_json is None:
                raise ValueError("FakeSupportModel expects a <context> block when drafting a reply")
            draft = render_reply(json.loads(context_json))
            message = AIMessage(content="", tool_calls=[make_tool_call(ReplyDraft.__name__, draft.model_dump())])
        elif tool_names:
            calls = [c for c in plan_lookups(messages) if c["name"] in tool_names]
            content = "" if calls else "I have gathered the facts needed for this ticket."
            message = AIMessage(content=content, tool_calls=calls)
        else:
            raise ValueError("FakeSupportModel only classifies, researches with tools and drafts structured replies")
        return ChatResult(generations=[ChatGeneration(message=message)])
