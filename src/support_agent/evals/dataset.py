"""Labelled evaluation tickets (JSON Lines, one case per line)."""

from __future__ import annotations

from itertools import chain, zip_longest
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

from support_agent.models import Intent, Resolution, TicketIn, Verdict


class Expected(BaseModel):
    intent: Intent
    verdict: Verdict
    resolution: Resolution
    kb_article: str | None = Field(
        default=None,
        description="For questions the help center answers: the article that should be retrieved and cited.",
    )


class EvalCase(BaseModel):
    id: str
    tests: str = Field(description="What the case checks, in one sentence.")
    ticket: TicketIn
    expected: Expected
    review: Literal["approve", "reject"] = Field(
        default="approve", description="The operator's decision if the agent pauses for approval."
    )


def load_dataset(path: Path) -> list[EvalCase]:
    lines = path.read_text(encoding="utf-8").splitlines()
    cases = [EvalCase.model_validate_json(line) for line in lines if line.strip()]
    duplicates = {c.id for c in cases if sum(other.id == c.id for other in cases) > 1}
    if duplicates:
        raise ValueError(f"Duplicate case ids in {path}: {sorted(duplicates)}")
    return cases


def stratified_sample(cases: list[EvalCase], size: int) -> list[EvalCase]:
    """A deterministic subset that covers every intent: round-robin over intents in file order.

    Used to run real models on a budget (`--sample 20`) without the skew of "the first N lines".
    """
    by_intent: dict[Intent, list[EvalCase]] = {}
    for case in cases:
        by_intent.setdefault(case.expected.intent, []).append(case)
    interleaved = (c for c in chain.from_iterable(zip_longest(*by_intent.values())) if c is not None)
    picked = {c.id for c in list(interleaved)[:size]}
    return [c for c in cases if c.id in picked]
