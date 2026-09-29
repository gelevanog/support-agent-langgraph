"""LLM provider abstraction (OpenAI, Anthropic, OpenRouter, deterministic fake) and prompts."""

from support_agent.llm.factory import build_chat_model, model_label, with_structured_output
from support_agent.llm.fake import FakeSupportModel

__all__ = ["FakeSupportModel", "build_chat_model", "model_label", "with_structured_output"]
