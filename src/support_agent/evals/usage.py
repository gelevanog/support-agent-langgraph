"""Counts LLM calls, tokens and (when the provider reports it) cost for the eval report."""

from __future__ import annotations

from typing import Any

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from pydantic import BaseModel


class ModelUsage(BaseModel):
    model: str
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    # USD as reported by the provider (OpenRouter returns it per response); None when unknown.
    cost_usd: float | None = None


class UsageCounter(BaseCallbackHandler):
    """Attach to a chat model (`model.callbacks = [counter]`) to meter every call it makes.

    Model-level callbacks survive `bind_tools` and `with_structured_output`, so the graph's
    classify, research and draft calls are all counted without touching the graph.
    """

    run_inline = True  # plain counters: run on the event loop, not in a thread pool

    def __init__(self, model: str) -> None:
        self.usage = ModelUsage(model=model)

    def on_chat_model_start(self, serialized: dict[str, Any], messages: list[list[BaseMessage]], **_: Any) -> None:
        self.usage.calls += 1

    def on_llm_end(self, response: LLMResult, **_: Any) -> None:
        for generation in (g for batch in response.generations for g in batch):
            if not isinstance(generation, ChatGeneration) or not isinstance(generation.message, AIMessage):
                continue
            metadata = generation.message.usage_metadata
            if metadata:
                self.usage.input_tokens += metadata["input_tokens"]
                self.usage.output_tokens += metadata["output_tokens"]
            cost = generation.message.response_metadata.get("token_usage", {}).get("cost")
            if isinstance(cost, int | float):
                self.usage.cost_usd = round((self.usage.cost_usd or 0.0) + cost, 6)
