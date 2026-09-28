"""Loader for the demo tickets in `examples/`."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel

from support_agent.models import TicketIn


class ExampleTicket(BaseModel):
    name: str
    title: str
    description: str
    ticket: TicketIn
    expected: dict[str, str]


def load_examples(directory: Path) -> list[ExampleTicket]:
    if not directory.is_dir():
        return []
    return [
        ExampleTicket.model_validate({"name": path.stem, **json.loads(path.read_text(encoding="utf-8"))})
        for path in sorted(directory.glob("*.json"))
    ]


def find_example(directory: Path, name: str) -> ExampleTicket:
    for example in load_examples(directory):
        if example.name == name or example.name.split("_", 1)[0] == name.zfill(2):
            return example
    raise KeyError(name)
