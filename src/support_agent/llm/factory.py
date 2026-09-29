"""Chat-model factory: one switch (`LLM_PROVIDER`) for OpenAI, Anthropic, OpenRouter or the offline fake."""

from __future__ import annotations

from typing import Any, TypeVar, cast

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from support_agent.config import Settings
from support_agent.llm.fake import FakeSupportModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Optional OpenRouter attribution headers (shown in the OpenRouter dashboard and app rankings).
OPENROUTER_HEADERS = {
    "HTTP-Referer": "https://github.com/gelevanog/support-agent-langgraph",
    "X-Title": "Support Autopilot",
}


def model_label(settings: Settings) -> str:
    """Provider and model, e.g. "openrouter/openai/gpt-5.4-mini" (for reports, health and logs)."""
    match settings.llm_provider:
        case "fake":
            return "fake"
        case "openai":
            return f"openai/{settings.openai_model}"
        case "anthropic":
            return f"anthropic/{settings.anthropic_model}"
        case "openrouter":
            return f"openrouter/{settings.openrouter_model}"


def build_chat_model(settings: Settings) -> BaseChatModel:
    match settings.llm_provider:
        case "fake":
            return FakeSupportModel()
        case "openai":
            from langchain_openai import ChatOpenAI

            openai_kwargs: dict[str, Any] = {
                "model": settings.openai_model,
                "max_tokens": settings.llm_max_tokens,
                "timeout": settings.llm_timeout_seconds,
                "max_retries": 2,
            }
            if settings.openai_api_key:
                openai_kwargs["api_key"] = settings.openai_api_key
            return ChatOpenAI(**openai_kwargs)
        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            # No temperature: current Claude models reject sampling parameters.
            anthropic_kwargs: dict[str, Any] = {
                "model": settings.anthropic_model,
                "max_tokens": settings.llm_max_tokens,
                "timeout": settings.llm_timeout_seconds,
                "max_retries": 2,
            }
            if settings.anthropic_api_key:
                anthropic_kwargs["api_key"] = settings.anthropic_api_key
            return ChatAnthropic(**anthropic_kwargs)
        case "openrouter":
            from langchain_openai import ChatOpenAI

            # OpenRouter speaks the OpenAI Chat Completions API (tools, JSON-schema output) for
            # every vendor, so the OpenAI client works with a different base URL and model id.
            openrouter_kwargs: dict[str, Any] = {
                "model": settings.openrouter_model,
                "base_url": settings.openrouter_base_url,
                "max_tokens": settings.llm_max_tokens,
                "timeout": settings.llm_timeout_seconds,
                "max_retries": 2,
                "use_responses_api": False,
                "default_headers": OPENROUTER_HEADERS,
            }
            if settings.openrouter_api_key:
                openrouter_kwargs["api_key"] = settings.openrouter_api_key
            return ChatOpenAI(**openrouter_kwargs)


def with_structured_output(llm: BaseChatModel, schema: type[SchemaT]) -> Runnable[LanguageModelInput, SchemaT]:
    """Structured output via the provider's native JSON-schema mode.

    Native schema-constrained decoding (OpenAI and OpenRouter `response_format`, Anthropic
    `output_config.format`) works together with reasoning/adaptive thinking, unlike forced tool
    calls. The fake model implements the generic tool-calling path.
    """
    runnable = llm.with_structured_output(schema, method="json_schema")
    return cast("Runnable[LanguageModelInput, SchemaT]", runnable)
