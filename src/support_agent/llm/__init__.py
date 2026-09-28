"""LLM provider abstraction (OpenAI, Anthropic, deterministic fake) and prompts."""

from support_agent.llm.factory import build_chat_model, with_structured_output
from support_agent.llm.fake import FakeSupportModel

__all__ = ["FakeSupportModel", "build_chat_model", "with_structured_output"]
