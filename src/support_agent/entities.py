"""Deterministic entity extraction used to validate / backfill what the LLM extracts."""

from __future__ import annotations

import re

ORDER_ID_RE = re.compile(r"(?:\border\b\s*(?:number|no\.?|id)?\s*[:#]?\s*|#)(\d{3,10})\b", re.IGNORECASE)
EMAIL_RE = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")
_ADDRESS_PATTERNS = (
    re.compile(
        r"new (?:shipping |delivery )?address(?: is|:)\s*(?P<addr>.+?)(?:[?!]|\.(?:\s|$)|\n|$)",
        re.IGNORECASE,
    ),
    re.compile(r"address\b[^.?!\n]*?\bto\s+(?P<addr>.+?)(?:[?!]|\.(?:\s|$)|\n|$)", re.IGNORECASE),
)


def extract_order_id(text: str) -> str | None:
    match = ORDER_ID_RE.search(text)
    return match.group(1) if match else None


def normalize_order_id(value: str | None) -> str | None:
    if value is None:
        return None
    digits = re.sub(r"\D", "", value)
    return digits or None


def extract_email(text: str) -> str | None:
    match = EMAIL_RE.search(text)
    return match.group(0).lower() if match else None


def extract_new_address(text: str) -> str | None:
    for pattern in _ADDRESS_PATTERNS:
        match = pattern.search(text)
        if match:
            address = match.group("addr").strip().rstrip(",;")
            if len(address) >= 5:
                return address
    return None
