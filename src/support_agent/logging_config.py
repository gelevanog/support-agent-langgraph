"""Structured logging via structlog (human-readable console or JSON lines)."""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def _stderr_logger(*_: Any) -> structlog.PrintLogger:
    return structlog.PrintLogger(file=sys.stderr)


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
    ]
    renderers: list[structlog.types.Processor] = (
        [structlog.processors.format_exc_info, structlog.processors.JSONRenderer()]
        if fmt == "json"
        else [
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty(), exception_formatter=structlog.dev.plain_traceback)
        ]
    )
    structlog.configure(
        processors=[*shared_processors, *renderers],
        wrapper_class=structlog.make_filtering_bound_logger(logging.getLevelName(level)),
        # Resolve sys.stderr per call so redirected/replaced streams (tests, CLIs) keep working.
        logger_factory=_stderr_logger,
        cache_logger_on_first_use=False,
    )
    logging.basicConfig(level=level, stream=sys.stderr, format="%(levelname)s %(name)s: %(message)s")
    # Third-party HTTP clients are chatty at INFO.
    for noisy in ("httpx", "httpx2", "httpcore", "openai", "anthropic"):
        logging.getLogger(noisy).setLevel(max(logging.getLevelName(level), logging.WARNING))


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]
