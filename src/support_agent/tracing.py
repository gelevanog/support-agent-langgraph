"""OpenTelemetry tracing: one trace per ticket run, with spans for graph nodes, LLM calls and tool calls.

Off by default. `OTEL_TRACES_EXPORTER=console` prints spans to stdout; `otlp` sends them to
`OTEL_EXPORTER_OTLP_ENDPOINT` (Jaeger, Grafana Tempo, Honeycomb, Langfuse, ...), which needs the
`tracing` extra. While disabled there is no tracer: `span()` returns a no-op context and no
LangChain callback is attached, so nothing is created, recorded or exported.

LLM and tool spans follow the OpenTelemetry GenAI semantic conventions (`gen_ai.*` attributes,
token usage included). LangSmith tracing is independent of this module and is switched on with
LangChain's own environment variables (`LANGSMITH_TRACING=true`, `LANGSMITH_API_KEY`).
"""

from __future__ import annotations

import sys
from collections.abc import Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from opentelemetry import trace
from opentelemetry.trace import Span, Status, StatusCode, Tracer

from support_agent import __version__
from support_agent.config import Settings

_tracer: Tracer | None = None
_callbacks: list[BaseCallbackHandler] = []


def configure_tracing(settings: Settings) -> None:
    """Install a tracer provider for the configured exporter. Idempotent; a no-op when disabled."""
    if settings.otel_traces_exporter == "none" or _tracer is not None:
        return
    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import SpanProcessor, TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter, SimpleSpanProcessor
    except ImportError as exc:  # pragma: no cover - depends on installed extras
        raise RuntimeError("OpenTelemetry export requires: uv sync --extra tracing") from exc

    processor: SpanProcessor
    if settings.otel_traces_exporter == "console":
        processor = SimpleSpanProcessor(ConsoleSpanExporter(out=sys.stdout))
    else:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

        processor = BatchSpanProcessor(OTLPSpanExporter())
    resource = Resource.create({"service.name": settings.otel_service_name, "service.version": __version__})
    provider = TracerProvider(resource=resource)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    use_tracer(provider.get_tracer("support_agent", __version__))


def use_tracer(tracer: Tracer | None) -> None:
    """Set (or with None, remove) the tracer used for all spans. Tests inject an in-memory one."""
    global _tracer, _callbacks
    _tracer = tracer
    _callbacks = [OpenTelemetryCallbackHandler(tracer)] if tracer is not None else []


def span(name: str, attributes: dict[str, Any] | None = None) -> AbstractContextManager[Span | None]:
    """Start a span as the current span, or do nothing when tracing is disabled.

    Exceptions are not recorded automatically (a LangGraph interrupt is control flow, not an
    error); call `record_error` for real failures.
    """
    if _tracer is None:
        return nullcontext()
    return _tracer.start_as_current_span(
        name, attributes=attributes, record_exception=False, set_status_on_exception=False
    )


def record_error(current: Span | None, exc: BaseException) -> None:
    if current is not None:
        current.record_exception(exc)
        current.set_status(Status(StatusCode.ERROR, f"{type(exc).__name__}: {exc}"))


def langchain_callbacks() -> list[BaseCallbackHandler]:
    """Callbacks to pass in the graph's RunnableConfig: empty unless tracing is enabled."""
    return list(_callbacks)


class OpenTelemetryCallbackHandler(BaseCallbackHandler):
    """Turns LangChain chat-model and tool runs into spans under the current (graph node) span."""

    # Called synchronously in the caller's context, so the node span is the parent.
    run_inline = True

    def __init__(self, tracer: Tracer) -> None:
        self._tracer = tracer
        self._spans: dict[UUID, Span] = {}

    def _end(self, run_id: UUID, error: BaseException | None = None) -> None:
        current = self._spans.pop(run_id, None)
        if current is not None:
            if error is not None:
                record_error(current, error)
            current.end()

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[BaseMessage]],
        *,
        run_id: UUID,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        meta = metadata or {}
        model = str(meta.get("ls_model_name") or kwargs.get("invocation_params", {}).get("model") or "unknown")
        attributes: dict[str, Any] = {
            "gen_ai.operation.name": "chat",
            "gen_ai.provider.name": str(meta.get("ls_provider", "unknown")),
            "gen_ai.request.model": model,
        }
        if meta.get("ls_max_tokens"):
            attributes["gen_ai.request.max_tokens"] = int(meta["ls_max_tokens"])
        if meta.get("langgraph_node"):
            attributes["support_agent.node"] = str(meta["langgraph_node"])
        self._spans[run_id] = self._tracer.start_span(f"chat {model}", attributes=attributes)

    def on_llm_end(self, response: LLMResult, *, run_id: UUID, **kwargs: Any) -> None:
        current = self._spans.get(run_id)
        if current is not None:
            generations = [g for batch in response.generations for g in batch if isinstance(g, ChatGeneration)]
            for generation in generations[:1]:
                message = generation.message
                usage = getattr(message, "usage_metadata", None)
                if usage:
                    current.set_attribute("gen_ai.usage.input_tokens", usage["input_tokens"])
                    current.set_attribute("gen_ai.usage.output_tokens", usage["output_tokens"])
                if model_name := message.response_metadata.get("model_name"):
                    current.set_attribute("gen_ai.response.model", str(model_name))
                if tool_calls := getattr(message, "tool_calls", None):
                    current.set_attribute("support_agent.tool_calls", [c["name"] for c in tool_calls])
        self._end(run_id)

    def on_llm_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error)

    def on_tool_start(
        self,
        serialized: dict[str, Any],
        input_str: str,
        *,
        run_id: UUID,
        inputs: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        name = str(serialized.get("name") or "tool")
        attributes: dict[str, Any] = {"gen_ai.operation.name": "execute_tool", "gen_ai.tool.name": name}
        if inputs:
            attributes["support_agent.tool.args"] = sorted(inputs)
        self._spans[run_id] = self._tracer.start_span(f"execute_tool {name}", attributes=attributes)

    def on_tool_end(self, output: Any, *, run_id: UUID, **kwargs: Any) -> None:
        current = self._spans.get(run_id)
        artifact = output.artifact if isinstance(output, ToolMessage) else None
        if current is not None and isinstance(artifact, dict):
            if "error" in artifact:
                current.set_status(Status(StatusCode.ERROR, str(artifact["error"])))
            if isinstance(articles := artifact.get("articles"), Sequence):
                current.set_attribute("support_agent.kb.article_ids", [str(a["id"]) for a in articles])
        self._end(run_id)

    def on_tool_error(self, error: BaseException, *, run_id: UUID, **kwargs: Any) -> None:
        self._end(run_id, error)
