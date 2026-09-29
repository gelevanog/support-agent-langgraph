"""Evaluation suite: dataset integrity, the deterministic judge, metrics and the CLI run."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_openai import ChatOpenAI
from typer.testing import CliRunner

from support_agent.config import Settings, get_settings
from support_agent.evals.cli import app
from support_agent.evals.dataset import EvalCase, Expected, load_dataset, stratified_sample
from support_agent.evals.fake_judge import FakeJudgeModel, grade
from support_agent.evals.judge import ReplyJudge, build_judge_model
from support_agent.evals.report import Rate, compute_metrics
from support_agent.evals.runner import CaseResult
from support_agent.evals.usage import UsageCounter
from support_agent.models import Intent, Resolution, Ticket, Verdict
from support_agent.store_api import StoreRepository
from tests.conftest import ROOT

DATASET = ROOT / "evals" / "tickets.jsonl"
runner = CliRunner()

CONTEXT = {
    "store_name": "Brewline Coffee",
    "resolution": "denied",
    "intent": "refund_request",
    "order": {"id": "1044", "total": "39.00", "delivered_on": "2026-08-15"},
    "reasons": ["Delivered 45 days ago, outside the 30-day refund window."],
    "kb_articles": [{"id": "KB-003", "title": "Returns and refund policy", "content": "Return within 30 days."}],
}
GOOD_REPLY = (
    "Hi David,\n\nThanks for reaching out. Unfortunately I can't refund order #1044: it was delivered on "
    "2026-08-15, outside our 30-day window.\n\nSee: Returns and refund policy [KB-003]\n\n"
    "Best regards,\nBrewline Coffee Support"
)


# --- Dataset ---------------------------------------------------------------------------------


def test_dataset_is_valid_and_covers_every_intent() -> None:
    cases = load_dataset(DATASET)
    assert 30 <= len(cases) <= 45
    assert {c.expected.intent for c in cases} == set(Intent)
    article_ids = {a.id for a in StoreRepository().list_knowledge_articles()}
    for case in cases:
        if case.expected.kb_article is not None:
            assert case.expected.intent is Intent.PRODUCT_QUESTION, case.id
            assert case.expected.kb_article in article_ids, case.id
    assert any(c.review == "reject" for c in cases)


def test_duplicate_case_ids_are_rejected(tmp_path: Path) -> None:
    line = DATASET.read_text(encoding="utf-8").splitlines()[0]
    path = tmp_path / "dup.jsonl"
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_dataset(path)


# --- Judge -----------------------------------------------------------------------------------


def test_fake_judge_passes_a_grounded_compliant_reply() -> None:
    judgement = grade(CONTEXT, GOOD_REPLY)
    assert (judgement.groundedness.score, judgement.tone.score, judgement.policy_compliance.score) == (5, 5, 5)
    assert judgement.passed


@pytest.mark.parametrize(
    ("reply", "criterion", "finding"),
    [
        (GOOD_REPLY.replace("#1044", "#1099"), "groundedness", "#1099"),
        (GOOD_REPLY.replace("[KB-003]", "[KB-009]"), "policy_compliance", "KB-009"),
        (GOOD_REPLY.replace("outside", "refund.within_window says outside"), "policy_compliance", "rule ids"),
        (
            GOOD_REPLY.replace("Unfortunately I can't", "I've issued a refund, but I can't"),
            "policy_compliance",
            "denied",
        ),
        (GOOD_REPLY.replace("Thanks for reaching out. Unfortunately I", "I"), "tone", "empathy"),
        (GOOD_REPLY.replace("Best regards,\nBrewline Coffee Support", "Bye"), "tone", "sign-off"),
    ],
)
def test_fake_judge_flags_problems(reply: str, criterion: str, finding: str) -> None:
    score = getattr(grade(CONTEXT, reply), criterion)
    assert score.score < 5
    assert finding in score.explanation


async def test_reply_judge_runs_through_structured_output() -> None:
    judge = ReplyJudge(FakeJudgeModel(), "Brewline Coffee")
    ticket = Ticket(id="T-1", body="Refund order #1044 please", customer_email="david.okafor@example.com")
    judgement = await judge.judge(ticket, CONTEXT, GOOD_REPLY)
    assert judgement.passed


def test_real_judge_models_come_from_the_provider_factory() -> None:
    settings = Settings(_env_file=None, openai_api_key="sk-test")  # type: ignore[call-arg]
    model = build_judge_model("openai", settings, "gpt-5")
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "gpt-5"
    assert isinstance(build_judge_model("fake", settings), FakeJudgeModel)


# --- Metrics ---------------------------------------------------------------------------------


def result(case_id: str, expected: tuple[Intent, str | None], got: str, retrieved: list[str]) -> CaseResult:
    intent, kb_article = expected
    return CaseResult(
        id=case_id,
        tests="",
        expected=Expected(intent=intent, verdict=Verdict.INFORM, resolution=Resolution.INFORMED, kb_article=kb_article),
        intent=got,
        verdict="inform",
        resolution="informed",
        retrieved_articles=retrieved,
        cited_articles=retrieved[:1],
    )


def test_metrics_confusion_and_retrieval_rates() -> None:
    metrics = compute_metrics(
        [
            result("a", (Intent.PRODUCT_QUESTION, "KB-001"), "product_question", ["KB-001", "KB-002"]),
            result("b", (Intent.PRODUCT_QUESTION, "KB-004"), "product_question", ["KB-003", "KB-004"]),
            result("c", (Intent.PRODUCT_QUESTION, "KB-005"), "refund_request", []),
            result("d", (Intent.ORDER_STATUS, None), "order_status", []),
        ]
    )
    assert metrics.confusion == {
        "order_status": {"order_status": 1},
        "product_question": {"product_question": 2, "refund_request": 1},
    }
    assert str(metrics.intent_accuracy) == "75.0% (3/4)"
    assert metrics.intent_recall["product_question"].value == pytest.approx(2 / 3, abs=1e-4)
    assert (metrics.kb_hit_rate.correct, metrics.kb_top1.correct, metrics.kb_cited.correct) == (2, 1, 1)
    assert metrics.all_checks_passed.correct == 3
    assert metrics.reply_quality.judged == 0


# --- CLI -------------------------------------------------------------------------------------


@pytest.fixture
def eval_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'unused.db'}")
    monkeypatch.setenv("COLUMNS", "160")
    get_settings.cache_clear()
    yield tmp_path
    get_settings.cache_clear()


def test_eval_cli_runs_the_whole_dataset_with_fake_models(eval_env: Path) -> None:
    output = eval_env / "report.json"
    args = ["--dataset", str(DATASET), "--output", str(output), "--fail-under", "0.85"]
    run = runner.invoke(app, args)
    assert run.exit_code == 0, run.output
    for heading in ("Summary", "Intent confusion", "Mismatches"):
        assert heading in run.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["agent_provider"] == "fake"
    assert report["judge"] == "fake"
    assert len(report["results"]) == len(load_dataset(DATASET))
    assert report["metrics"]["errors"] == 0
    assert report["metrics"]["kb_hit_rate"]["total"] == sum(1 for c in load_dataset(DATASET) if c.expected.kb_article)
    assert all(r["judgement"] for r in report["results"])


@pytest.mark.parametrize(("case_id", "exit_code"), [("refund-chipped-auto", 0), ("faq-refund-timing", 1)])
def test_fail_under_gates_on_accuracy(eval_env: Path, case_id: str, exit_code: int) -> None:
    case = next(c for c in load_dataset(DATASET) if c.id == case_id)
    dataset = eval_env / "one.jsonl"
    dataset.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    args = ["--dataset", str(dataset), "--output", str(eval_env / "r.json"), "--fail-under", "1.0"]
    run = runner.invoke(app, args)
    assert run.exit_code == exit_code, run.output


def test_eval_case_defaults_to_approving() -> None:
    case = EvalCase.model_validate(
        {
            "id": "x",
            "tests": "t",
            "ticket": {"body": "Where is my order #1045?"},
            "expected": {"intent": "order_status", "verdict": "inform", "resolution": "informed"},
        }
    )
    assert case.review == "approve"


def test_stratified_sample_covers_every_intent_deterministically() -> None:
    cases = load_dataset(DATASET)
    sample = stratified_sample(cases, 20)
    assert len(sample) == 20
    assert {c.expected.intent for c in sample} == set(Intent)
    assert [c.id for c in sample] == [c.id for c in stratified_sample(cases, 20)]
    assert [c.id for c in sample] == [c.id for c in cases if c in sample]  # file order is kept
    assert any(c.expected.kb_article is None and c.expected.intent is Intent.PRODUCT_QUESTION for c in sample)
    assert stratified_sample(cases, 1000) == cases


def test_eval_cli_sample_and_usage_accounting(eval_env: Path) -> None:
    output = eval_env / "report.json"
    run = runner.invoke(app, ["--dataset", str(DATASET), "--output", str(output), "--sample", "6"])
    assert run.exit_code == 0, run.output
    assert "LLM usage" in run.output
    report = json.loads(output.read_text(encoding="utf-8"))
    assert {r["expected"]["intent"] for r in report["results"]} == {i.value for i in Intent}
    usage = report["usage"]
    assert usage["agent"]["model"] == "fake"
    # classify + research rounds + draft per ticket; one judge call per reply
    assert usage["agent"]["calls"] >= 3 * 6
    assert usage["judge"]["calls"] == sum(1 for r in report["results"] if r["reply"])
    assert usage["agent"]["cost_usd"] is None


def test_usage_counter_sums_tokens_and_provider_reported_cost() -> None:
    counter = UsageCounter("openrouter/openai/gpt-5.4-mini")
    message = AIMessage(
        content="ok",
        usage_metadata={"input_tokens": 1200, "output_tokens": 80, "total_tokens": 1280},
        response_metadata={"token_usage": {"cost": 0.00125}},
    )
    for _ in range(2):
        counter.on_chat_model_start({}, [[HumanMessage(content="hi")]])
        counter.on_llm_end(LLMResult(generations=[[ChatGeneration(message=message)]]))
    assert counter.usage.model_dump() == {
        "model": "openrouter/openai/gpt-5.4-mini",
        "calls": 2,
        "input_tokens": 2400,
        "output_tokens": 160,
        "cost_usd": 0.0025,
    }


def test_rate_without_cases_renders_as_not_applicable() -> None:
    assert str(Rate.of([])) == "n/a (no cases)"
    assert str(Rate.of([True, False])) == "50.0% (1/2)"
