from __future__ import annotations

from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from support_agent.config import Settings
from support_agent.examples import ExampleTicket, load_examples
from support_agent.runtime import Runtime, create_runtime

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = load_examples(ROOT / "examples")


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,  # type: ignore[call-arg]
        llm_provider="fake",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'test.db'}",
        examples_dir=ROOT / "examples",
        store_api_url=None,
        log_level="WARNING",
    )


@pytest.fixture
async def runtime(settings: Settings) -> AsyncIterator[Runtime]:
    async with create_runtime(settings) as rt:
        yield rt


def example(name: str) -> ExampleTicket:
    return next(e for e in EXAMPLES if e.name.startswith(name))
