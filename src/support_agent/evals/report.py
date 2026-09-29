"""Metrics over case results, the JSON report and its terminal rendering (rich)."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from statistics import mean

from pydantic import BaseModel
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from support_agent.evals.judge import PASSING_SCORE
from support_agent.evals.runner import CaseResult
from support_agent.evals.usage import ModelUsage
from support_agent.models import Intent

CRITERIA = ("groundedness", "tone", "policy_compliance")


class Rate(BaseModel):
    correct: int
    total: int
    value: float

    @classmethod
    def of(cls, flags: list[bool]) -> Rate:
        correct, total = sum(flags), len(flags)
        return cls(correct=correct, total=total, value=round(correct / total, 4) if total else 0.0)

    def __str__(self) -> str:
        return f"{self.value:.1%} ({self.correct}/{self.total})" if self.total else "n/a (no cases)"


class ReplyQuality(BaseModel):
    judged: int
    mean_scores: dict[str, float]
    passing: Rate  # replies with every criterion >= PASSING_SCORE


class EvalMetrics(BaseModel):
    intent_accuracy: Rate
    intent_recall: dict[str, Rate]
    confusion: dict[str, dict[str, int]]  # expected intent -> predicted intent -> count
    verdict_accuracy: Rate
    resolution_accuracy: Rate
    kb_hit_rate: Rate  # expected article among the retrieved (top 3)
    kb_top1: Rate
    kb_cited: Rate  # expected article cited in the reply
    reply_quality: ReplyQuality
    all_checks_passed: Rate  # intent, verdict, resolution and (if expected) KB hit all correct
    errors: int


class EvalReport(BaseModel):
    generated_at: datetime
    dataset: str
    agent_provider: str
    judge: str
    embeddings: str
    kb_backend: str
    usage: dict[str, ModelUsage] = {}  # "agent" / "judge" -> calls, tokens, cost
    metrics: EvalMetrics
    results: list[CaseResult]


def compute_metrics(results: list[CaseResult]) -> EvalMetrics:
    confusion: dict[str, dict[str, int]] = {}
    for intent in Intent:
        predicted = Counter(r.intent or "error" for r in results if r.expected.intent is intent)
        if predicted:
            confusion[intent.value] = dict(predicted)
    kb_cases = [r for r in results if r.expected.kb_article is not None]
    judged = [r.judgement for r in results if r.judgement is not None]
    return EvalMetrics(
        intent_accuracy=Rate.of([r.intent_ok for r in results]),
        intent_recall={
            intent: Rate.of([r.intent_ok for r in results if r.expected.intent == intent]) for intent in confusion
        },
        confusion=confusion,
        verdict_accuracy=Rate.of([r.verdict_ok for r in results]),
        resolution_accuracy=Rate.of([r.resolution_ok for r in results]),
        kb_hit_rate=Rate.of([bool(r.kb_hit) for r in kb_cases]),
        kb_top1=Rate.of([r.retrieved_articles[:1] == [r.expected.kb_article] for r in kb_cases]),
        kb_cited=Rate.of([r.expected.kb_article in r.cited_articles for r in kb_cases]),
        reply_quality=ReplyQuality(
            judged=len(judged),
            mean_scores={c: round(mean(getattr(j, c).score for j in judged), 2) if judged else 0.0 for c in CRITERIA},
            passing=Rate.of([j.passed for j in judged]),
        ),
        all_checks_passed=Rate.of(
            [r.intent_ok and r.verdict_ok and r.resolution_ok and r.kb_hit is not False for r in results]
        ),
        errors=sum(r.error is not None for r in results),
    )


def _style(rate: Rate) -> str:
    if not rate.total:
        return "dim"
    return "green" if rate.value >= 0.9 else "yellow" if rate.value >= 0.75 else "red"


def _summary_table(report: EvalReport) -> Table:
    m = report.metrics
    table = Table(title="Summary", title_justify="left")
    table.add_column("Metric")
    table.add_column("Score", justify="right")
    rows: list[tuple[str, Rate]] = [
        ("Intent accuracy", m.intent_accuracy),
        ("Decision (verdict) accuracy", m.verdict_accuracy),
        ("Action (resolution) accuracy", m.resolution_accuracy),
        ("KB hit rate (expected article in top 3)", m.kb_hit_rate),
        ("KB top-1 accuracy", m.kb_top1),
        ("KB article cited in reply", m.kb_cited),
        ("Cases with every check correct", m.all_checks_passed),
        (f"Replies scoring >= {PASSING_SCORE} on every criterion", m.reply_quality.passing),
    ]
    for label, rate in rows:
        table.add_row(label, Text(str(rate), style=_style(rate)))
    for criterion, score in m.reply_quality.mean_scores.items():
        table.add_row(f"Judge: {criterion.replace('_', ' ')} (mean, 1-5)", f"{score:.2f}")
    return table


def _confusion_table(metrics: EvalMetrics) -> Table:
    predicted = sorted({p for row in metrics.confusion.values() for p in row})
    table = Table(title="Intent confusion (rows: expected, columns: predicted)", title_justify="left")
    table.add_column("expected")
    for column in predicted:
        table.add_column(column.replace("_", " "), justify="right")
    table.add_column("recall", justify="right")
    for expected, row in metrics.confusion.items():
        cells = [
            Text(str(row.get(p, "")), style="bold green" if p == expected else "red") if row.get(p) else Text("")
            for p in predicted
        ]
        recall = metrics.intent_recall[expected]
        table.add_row(expected.replace("_", " "), *cells, Text(f"{recall.value:.0%}", style=_style(recall)))
    return table


def _failures_table(results: list[CaseResult]) -> Table | None:
    table = Table(title="Mismatches", title_justify="left")
    for column in ("Case", "Check", "Expected", "Got"):
        table.add_column(column)
    for r in results:
        checks = [
            ("intent", r.intent_ok, r.expected.intent.value, r.intent),
            ("verdict", r.verdict_ok, r.expected.verdict.value, r.verdict),
            ("resolution", r.resolution_ok, r.expected.resolution.value, r.resolution),
            ("kb article", r.kb_hit is not False, r.expected.kb_article, ", ".join(r.retrieved_articles) or "none"),
        ]
        for check, ok, expected, got in checks:
            if not ok:
                table.add_row(r.id, check, str(expected), (r.error and "error") or str(got))
    return table if table.row_count else None


def _judge_table(results: list[CaseResult]) -> Table | None:
    table = Table(title="Judge findings (every criterion scored below 5)", title_justify="left")
    for column in ("Case", "Criterion", "Score", "Explanation"):
        table.add_column(column)
    for r in results:
        if r.judgement is None:
            continue
        for criterion in CRITERIA:
            score = getattr(r.judgement, criterion)
            if score.score < 5:
                table.add_row(r.id, criterion.replace("_", " "), str(score.score), score.explanation)
    return table if table.row_count else None


def _usage_table(usage: dict[str, ModelUsage]) -> Table | None:
    table = Table(title="LLM usage", title_justify="left")
    for column in ("Role", "Model", "Calls", "Input tokens", "Output tokens", "Cost (USD)"):
        table.add_column(column, justify="left" if column in ("Role", "Model") else "right")
    for role, u in usage.items():
        if u.calls:
            cost = "-" if u.cost_usd is None else f"{u.cost_usd:.4f}"
            table.add_row(role, u.model, str(u.calls), f"{u.input_tokens:,}", f"{u.output_tokens:,}", cost)
    return table if table.row_count else None


def render(report: EvalReport, console: Console) -> None:
    header = (
        f"{len(report.results)} cases from {report.dataset}\n"
        f"agent: {report.agent_provider}   judge: {report.judge}   "
        f"retrieval: {report.embeddings} ({report.kb_backend})"
    )
    console.print(Panel(header, title="Support Autopilot evaluation", expand=False))
    tables = [
        _summary_table(report),
        _confusion_table(report.metrics),
        _failures_table(report.results),
        _judge_table(report.results),
        _usage_table(report.usage),
    ]
    console.print(Group(*(t for t in tables if t is not None)))
    if report.metrics.errors:
        console.print(Text(f"{report.metrics.errors} case(s) failed with an error; see the JSON report.", style="red"))
