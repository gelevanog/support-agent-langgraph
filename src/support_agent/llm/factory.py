"""Chat-model factory: one switch (`LLM_PROVIDER`) for OpenAI, Anthropic or the offline fake."""

from __future__ import annotations

from typing import Any, TypeVar, cast

from langchain_core.language_models import BaseChatModel, LanguageModelInput
from langchain_core.runnables import Runnable
from pydantic import BaseModel

from support_agent.config import Settings
from support_agent.llm.fake import FakeSupportModel

SchemaT = TypeVar("SchemaT", bound=BaseModel)


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


def with_structured_output(llm: BaseChatModel, schema: type[SchemaT]) -> Runnable[LanguageModelInput, SchemaT]:
    """Structured output via the provider's native JSON-schema mode.

    Native schema-constrained decoding (OpenAI `response_format`, Anthropic
    `output_config.format`) works together with reasoning/adaptive thinking, unlike forced tool
    calls. The fake model implements the generic tool-calling path.
    """
    runnable = llm.with_structured_output(schema, method="json_schema")
    return cast("Runnable[LanguageModelInput, SchemaT]", runnable)
