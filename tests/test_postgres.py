"""Optional: the pause/resume flow against Postgres (tickets table + AsyncPostgresSaver).

Runs only when TEST_POSTGRES_URL is set, e.g.
    TEST_POSTGRES_URL=postgresql+psycopg://support:support@localhost:5432/support uv run pytest tests/test_postgres.py
"""

from __future__ import annotations

import os
from decimal import Decimal

import pytest

from support_agent.config import Settings
from support_agent.models import TicketStatus
from support_agent.runtime import create_runtime
from tests.conftest import ROOT, example

POSTGRES_URL = os.environ.get("TEST_POSTGRES_URL")

pytestmark = pytest.mark.skipif(not POSTGRES_URL, reason="TEST_POSTGRES_URL not set")


async def test_pause_and_resume_on_postgres() -> None:
    settings = Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider="fake",
        database_url=POSTGRES_URL or "",
        examples_dir=ROOT / "examples",
        log_level="WARNING",
    )
    async with create_runtime(settings) as first:
        paused = await first.service.submit(example("02").ticket)
        assert paused.status is TicketStatus.AWAITING_APPROVAL
    async with create_runtime(settings) as second:
        done = await second.service.approve(paused.id, operator="maria")
        assert done.resolution == "refund_issued"
        assert second.store.refunds[0].amount == Decimal("249.00")
