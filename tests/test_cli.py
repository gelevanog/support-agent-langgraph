"""CLI smoke tests (fake provider, temporary database)."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path

import pytest
from typer.testing import CliRunner

from support_agent.cli import app
from support_agent.config import get_settings
from tests.conftest import ROOT

runner = CliRunner()


@pytest.fixture(autouse=True)
def cli_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    monkeypatch.setenv("LLM_PROVIDER", "fake")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'cli.db'}")
    monkeypatch.setenv("EXAMPLES_DIR", str(ROOT / "examples"))
    monkeypatch.setenv("COLUMNS", "160")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_run_prints_trace_and_reply() -> None:
    result = runner.invoke(app, ["run", "Where is my order #1045?", "--email", "emma.rossi@example.com"])
    assert result.exit_code == 0, result.output
    assert "get_order" in result.output
    assert "order.identified" in result.output
    assert "Draft reply" in result.output
    assert "1Z999AA10198765432" in result.output


def test_pause_then_approve_in_a_separate_invocation() -> None:
    first = runner.invoke(app, ["example", "02"])
    assert first.exit_code == 0, first.output
    assert "Paused for human approval" in first.output
    ticket_id = re.search(r"approve (TCK-[0-9A-F]+)", first.output).group(1)  # type: ignore[union-attr]

    listed = runner.invoke(app, ["list", "--status", "awaiting_approval"])
    assert ticket_id in listed.output

    second = runner.invoke(app, ["approve", ticket_id, "--operator", "maria"])
    assert second.exit_code == 0, second.output
    assert "Approved by maria" in second.output
    assert "refund_issued" in second.output

    again = runner.invoke(app, ["approve", ticket_id])
    assert again.exit_code == 1
    assert "not awaiting approval" in again.output


def test_example_with_inline_reject() -> None:
    result = runner.invoke(app, ["example", "02", "--reject", "--note", "too many returns"])
    assert result.exit_code == 0, result.output
    assert "rejected_after_review" in result.output


def test_demo_runs_all_examples() -> None:
    result = runner.invoke(app, ["demo"])
    assert result.exit_code == 0, result.output
    for resolution in ("refund_issued", "denied", "informed", "escalated", "address_updated"):
        assert resolution in result.output


def test_show_unknown_ticket() -> None:
    result = runner.invoke(app, ["show", "TCK-NOPE"])
    assert result.exit_code == 1
    assert "not found" in result.output


def test_graph_prints_mermaid() -> None:
    result = runner.invoke(app, ["graph"])
    assert result.exit_code == 0
    assert "apply_policy -.-> human_approval" in result.output
