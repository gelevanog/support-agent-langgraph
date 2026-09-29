"""Deterministic judge for tests and CI: checkable proxies for groundedness, tone and policy compliance.

It implements the same structured-output interface as a real judge model, so `ReplyJudge` runs
unchanged. The checks are mechanical (every amount, reference, order number, date and tracking
number in the reply must appear in the context; no leaked rule ids or unauthorised refund
promises; valid citations; greeting, sign-off, length, empathy when declining). They catch
regressions, not subtle quality problems: that is what a model judge is for.
"""

from __future__ import annotations

import json
import re
from typing import Any, ClassVar

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from support_agent.evals.judge import CriterionScore, ReplyJudgement, Score
from support_agent.llm.fake import OfflineChatModel, extract_tag, make_tool_call

_FACTS = re.compile(
    r"\$\d[\d,]*(?:\.\d{2})?"  # amounts
    r"|\b(?:RF|ESC|KB)-[0-9A-F]+\b"  # refund, escalation and article references
    r"|#\d{3,}"  # order numbers
    r"|\b\d{4}-\d{2}-\d{2}\b"  # dates
    r"|\b(?=[A-Z0-9]*\d)(?=[A-Z0-9]*[A-Z])[A-Z0-9]{10,}\b"  # tracking numbers
)
_RULE_ID = re.compile(r"\b(?:order|identity|refund|address|kb|complaint|fallback)\.[a-z_]+\b")
_REFUND_PROMISE = re.compile(r"\b(?:issued|processed)\b[^.]*\brefund\b|\brefund of \$", re.IGNORECASE)
_FOLLOW_UP = re.compile(r"get back to you|follow up|in touch|specialist|our team", re.IGNORECASE)
_EMPATHY = re.compile(r"sorry|unfortunately|apolog|understand|thank", re.IGNORECASE)
_CITATION = re.compile(r"^See: .*\[(KB-\d+)\]$", re.MULTILINE)
_SHOUTING = re.compile(r"\b[A-Z]{4,}\b")
_BAD_NEWS = {"denied", "rejected_after_review"}


def _score(penalty: int) -> Score:
    scores: tuple[Score, ...] = (5, 4, 3, 2, 1)
    return scores[min(penalty, 4)]


def _criterion(issues: list[tuple[str, int]], ok: str) -> CriterionScore:
    if not issues:
        return CriterionScore(explanation=ok, score=5)
    return CriterionScore(
        explanation=" ".join(f"{issue}." for issue, _ in issues), score=_score(sum(weight for _, weight in issues))
    )


def grade(context: dict[str, Any], reply: str) -> ReplyJudgement:
    context_text = json.dumps(context, ensure_ascii=False)
    resolution = context["resolution"]
    kb_ids = {a["id"] for a in context.get("kb_articles", [])}
    body = reply.split("\n\nSee: ")[0].split("\n\nBest regards")[0]

    facts = list(dict.fromkeys(_FACTS.findall(reply)))
    unsupported = [f for f in facts if f.lstrip("$#").replace(",", "") not in context_text]
    grounded: list[tuple[str, int]] = []
    if unsupported:
        grounded.append((f"Not supported by the context: {', '.join(unsupported)}", 2 + len(unsupported)))

    policy: list[tuple[str, int]] = []
    if leaked := _RULE_ID.findall(reply):
        policy.append((f"Reveals internal rule ids: {', '.join(leaked)}", 3))
    if resolution != "refund_issued" and _REFUND_PROMISE.search(body):
        policy.append((f"Mentions an issued refund although the resolution is {resolution}", 3))
    cited = _CITATION.findall(reply)
    if invalid := [c for c in cited if c not in kb_ids]:
        policy.append((f"Cites articles that are not in the context: {', '.join(invalid)}", 3))
    if resolution == "informed" and context.get("intent") == "product_question" and kb_ids and not cited:
        policy.append(("Answers from a help-center article without citing it", 1))
    if resolution == "escalated" and not _FOLLOW_UP.search(body):
        policy.append(("Escalated, but the reply does not say that the team will follow up", 1))
    if resolution == "need_more_info" and "?" not in body:
        policy.append(("Needs more information, but the reply asks no question", 1))

    tone: list[tuple[str, int]] = []
    if not re.match(r"(Hi|Hello|Dear)\b", reply):
        tone.append(("No greeting", 1))
    if "Support" not in reply.rsplit("\n", 1)[-1]:
        tone.append(("No sign-off", 1))
    if (words := len(body.split())) > 150:
        tone.append((f"Too long ({words} words, limit 150)", 1))
    if len([w for w in _SHOUTING.findall(body) if w not in facts]) >= 2:
        tone.append(("Uses all-caps words", 1))
    if resolution in _BAD_NEWS and not _EMPATHY.search(body):
        tone.append(("Declines the request without any empathy", 1))

    return ReplyJudgement(
        groundedness=_criterion(
            grounded,
            f"Every concrete fact ({', '.join(facts)}) appears in the context."
            if facts
            else "No amounts, references, dates or tracking numbers to verify.",
        ),
        tone=_criterion(tone, "Greets, signs off, stays concise and matches the situation."),
        policy_compliance=_criterion(
            policy, f"Consistent with the '{resolution}' resolution; no internal details; citations valid."
        ),
    )


class FakeJudgeModel(OfflineChatModel):
    """Answers `ReplyJudgement` structured-output requests with the rule-based `grade()`."""

    model_id: ClassVar[str] = "fake-judge"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        prompt = self.last_human_text(messages)
        context_json, reply = extract_tag(prompt, "context"), extract_tag(prompt, "reply")
        if ReplyJudgement.__name__ not in self.bound_tool_names(kwargs) or context_json is None or reply is None:
            raise ValueError("FakeJudgeModel only answers ReplyJudgement requests with <context> and <reply>")
        judgement = grade(json.loads(context_json), reply)
        call = make_tool_call(ReplyJudgement.__name__, judgement.model_dump())
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content="", tool_calls=[call]))])
