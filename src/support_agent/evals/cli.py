"""Eval command: run the labelled tickets through the agent and report the scores.

python -m support_agent.evals                                  # fake agent + fake judge, no keys
python -m support_agent.evals --provider anthropic --judge openai --judge-model gpt-5
python -m support_agent.evals --provider openrouter --judge openrouter --judge-model anthropic/claude-sonnet-5
"""

from __future__ import annotations

import asyncio
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.text import Text
from sqlalchemy.ext.asyncio import create_async_engine

from support_agent.config import LLMProvider, Settings, get_settings
from support_agent.evals.dataset import load_dataset, stratified_sample
from support_agent.evals.judge import ReplyJudge, build_judge_model, judge_settings
from support_agent.evals.report import EvalReport, compute_metrics, render
from support_agent.evals.runner import CaseResult, run_suite
from support_agent.evals.usage import UsageCounter
from support_agent.knowledge import build_knowledge_base
from support_agent.llm import build_chat_model, model_label
from support_agent.logging_config import configure_logging
from support_agent.store_api import StoreRepository
from support_agent.tracing import configure_tracing

# No locals in tracebacks: they would print client objects that hold API keys.
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)
console = Console(highlight=False)


def _embeddings_label(settings: Settings) -> str:
    model = {"openai": settings.openai_embeddings_model, "openrouter": settings.openrouter_embeddings_model}
    name: str = settings.embeddings_provider
    if name in model:
        name += f"/{model[name]}"
    return f"{name}/{settings.embeddings_dimensions}d"


def _progress(result: CaseResult) -> None:
    ok = result.intent_ok and result.verdict_ok and result.resolution_ok and result.kb_hit is not False
    console.print(Text(f"  {'ok' if ok else 'x':<3}{result.id}", style="dim" if ok else "yellow"))


@app.command()
def main(
    dataset: Annotated[Path, typer.Option(help="JSON Lines file with labelled tickets.")] = Path("evals/tickets.jsonl"),
    provider: Annotated[LLMProvider | None, typer.Option(help="Agent LLM provider (default: LLM_PROVIDER).")] = None,
    judge: Annotated[LLMProvider, typer.Option(help="Judge provider; fake = deterministic rule-based judge.")] = "fake",
    judge_model: Annotated[str | None, typer.Option(help="Judge model (default: the provider's model).")] = None,
    output: Annotated[Path, typer.Option(help="Where to write the JSON report.")] = Path("data/eval-report.json"),
    concurrency: Annotated[int, typer.Option(min=1, max=32, help="Cases run in parallel.")] = 4,
    limit: Annotated[int | None, typer.Option(min=1, help="Only run the first N cases.")] = None,
    case: Annotated[list[str] | None, typer.Option("--case", help="Only run this case id (repeatable).")] = None,
    sample: Annotated[
        int | None, typer.Option(min=1, help="Run a deterministic N-case subset covering every intent.")
    ] = None,
    fail_under: Annotated[
        float, typer.Option(min=0, max=1, help="Exit 1 if intent, verdict or resolution accuracy is below this.")
    ] = 0.0,
) -> None:
    """Evaluate classification, decisions, retrieval and reply quality on a labelled dataset."""
    settings = get_settings()
    if provider is not None:
        settings = settings.model_copy(update={"llm_provider": provider})
    configure_logging("WARNING", settings.log_format)
    configure_tracing(settings)
    cases = load_dataset(dataset)[:limit]
    if case:
        unknown = set(case) - {c.id for c in cases}
        if unknown:
            raise typer.BadParameter(f"unknown case id(s): {', '.join(sorted(unknown))}", param_hint="--case")
        cases = [c for c in cases if c.id in case]
    if sample is not None:
        cases = stratified_sample(cases, sample)
    agent_usage = UsageCounter(model_label(settings))
    judge_usage = UsageCounter(model_label(judge_settings(judge, settings, judge_model)))

    async def evaluate() -> list[CaseResult]:
        engine = create_async_engine(settings.database_url)  # only used by KB_BACKEND=pgvector
        try:
            knowledge_base = build_knowledge_base(settings, engine)
            await knowledge_base.index(StoreRepository().list_knowledge_articles())
            agent_llm = build_chat_model(settings)
            agent_llm.callbacks = [agent_usage]
            judge_llm = build_judge_model(judge, settings, judge_model)
            judge_llm.callbacks = [judge_usage]
            reply_judge = ReplyJudge(judge_llm, settings.store_name)
            with tempfile.TemporaryDirectory(prefix="support-evals-") as workdir:
                return await run_suite(
                    cases,
                    settings,
                    agent_llm,
                    knowledge_base,
                    reply_judge,
                    Path(workdir),
                    concurrency=concurrency,
                    on_result=_progress,
                )
        finally:
            await engine.dispose()

    results = asyncio.run(evaluate())
    report = EvalReport(
        generated_at=datetime.now(UTC),
        dataset=str(dataset),
        agent_provider=agent_usage.usage.model,
        judge=judge_usage.usage.model,
        embeddings=_embeddings_label(settings),
        kb_backend=settings.kb_backend,
        usage={"agent": agent_usage.usage, "judge": judge_usage.usage},
        metrics=compute_metrics(results),
        results=results,
    )
    render(report, console)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    console.print(f"\nJSON report: {output}")

    m = report.metrics
    below = [r for r in (m.intent_accuracy, m.verdict_accuracy, m.resolution_accuracy) if r.value < fail_under]
    if below or m.errors:
        raise typer.Exit(code=1)
