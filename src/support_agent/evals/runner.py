"""Runs every case through the real agent (graph, rules, tools, checkpointer) and judges the reply.

Each case gets a fresh mock store and its own SQLite file, so refunds or address changes made by
one case never leak into another. The embedding index is built once and shared.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from pathlib import Path
from typing import Any

from langchain_core.language_models import BaseChatModel
from pydantic import BaseModel

from support_agent.config import Settings
from support_agent.evals.dataset import EvalCase, Expected
from support_agent.evals.judge import ReplyJudge, ReplyJudgement
from support_agent.knowledge import KnowledgeBase
from support_agent.models import AuditKind, Ticket, TicketStatus
from support_agent.runtime import create_runtime
from support_agent.service import TicketView
from support_agent.store_api import StoreRepository


class CaseResult(BaseModel):
    id: str
    tests: str
    expected: Expected
    intent: str | None = None
    verdict: str | None = None
    resolution: str | None = None
    status: str | None = None
    paused_for_review: bool = False
    retrieved_articles: list[str] = []
    cited_articles: list[str] = []
    reply: str | None = None
    judgement: ReplyJudgement | None = None
    error: str | None = None

    @property
    def intent_ok(self) -> bool:
        return self.intent == self.expected.intent

    @property
    def verdict_ok(self) -> bool:
        return self.verdict == self.expected.verdict

    @property
    def resolution_ok(self) -> bool:
        return self.resolution == self.expected.resolution

    @property
    def kb_hit(self) -> bool | None:
        """Expected article among the retrieved ones (None when the case expects no article)."""
        expected = self.expected.kb_article
        return None if expected is None else expected in self.retrieved_articles


def _reply_context(view: TicketView) -> dict[str, Any]:
    return next(e.data["context"] for e in reversed(view.audit) if e.kind is AuditKind.REPLY)


async def run_case(
    case: EvalCase,
    settings: Settings,
    llm: BaseChatModel,
    knowledge_base: KnowledgeBase,
    judge: ReplyJudge,
    workdir: Path,
) -> CaseResult:
    case_settings = settings.model_copy(update={"database_url": f"sqlite+aiosqlite:///{workdir / case.id}.db"})
    result = CaseResult(id=case.id, tests=case.tests, expected=case.expected)
    async with create_runtime(case_settings, llm=llm, store=StoreRepository(), knowledge_base=knowledge_base) as rt:
        view = await rt.service.submit(case.ticket)
        if view.status is TicketStatus.AWAITING_APPROVAL:
            result.paused_for_review = True
            review = rt.service.approve if case.review == "approve" else rt.service.reject
            view = await review(view.id, operator="eval-operator", note=f"Eval case {case.id}")
    result.intent, result.verdict, result.resolution = view.intent, view.verdict, view.resolution
    result.status, result.reply, result.error = view.status.value, view.reply, view.error
    if view.knowledge is not None:
        result.retrieved_articles = [a.id for a in view.knowledge.articles]
        result.cited_articles = [a.id for a in view.knowledge.articles if a.cited]
    if view.reply:
        ticket = Ticket(id=view.id, received_at=view.created_at, **case.ticket.model_dump())
        result.judgement = await judge.judge(ticket, _reply_context(view), view.reply)
    return result


async def run_suite(
    cases: list[EvalCase],
    settings: Settings,
    llm: BaseChatModel,
    knowledge_base: KnowledgeBase,
    judge: ReplyJudge,
    workdir: Path,
    *,
    concurrency: int = 4,
    on_result: Callable[[CaseResult], None] | None = None,
) -> list[CaseResult]:
    semaphore = asyncio.Semaphore(concurrency)

    async def run_one(case: EvalCase) -> CaseResult:
        async with semaphore:
            try:
                result = await run_case(case, settings, llm, knowledge_base, judge, workdir)
            except Exception as exc:  # a crashing case is a result, not the end of the run
                result = CaseResult(id=case.id, tests=case.tests, expected=case.expected, error=repr(exc))
            if on_result is not None:
                on_result(result)
            return result

    return list(await asyncio.gather(*(run_one(case) for case in cases)))
