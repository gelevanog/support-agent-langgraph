"""Deterministic, configurable business rules (no LLM involved)."""

from support_agent.rules.policy import PolicyConfig, evaluate

__all__ = ["PolicyConfig", "evaluate"]
