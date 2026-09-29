"""OpenTelemetry spans for ticket runs, graph nodes, LLM calls and tool calls."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from support_agent import tracing
from support_agent.config import Settings
from support_agent.runtime import create_runtime
from tests.conftest import example
from tests.test_graph import FailingModel

sdk_trace = pytest.importorskip("opentelemetry.sdk.trace")
in_memory = pytest.importorskip("opentelemetry.sdk.trace.export.in_memory_span_exporter")
from opentelemetry.sdk.trace import ReadableSpan  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.trace import StatusCode  # noqa: E402


@pytest.fixture
def exporter() -> Iterator[in_memory.InMemorySpanExporter]:
    span_exporter = in_memory.InMemorySpanExporter()
    provider = sdk_trace.TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    tracing.use_tracer(provider.get_tracer("test"))
    yield span_exporter
    tracing.use_tracer(None)


def children(spans: tuple[ReadableSpan, ...], parent: ReadableSpan) -> list[ReadableSpan]:
    assert parent.context is not None
    return [s for s in spans if s.parent is not None and s.parent.span_id == parent.context.span_id]


def one(spans: tuple[ReadableSpan, ...], name: str) -> ReadableSpan:
    matches = [s for s in spans if s.name == name]
    assert len(matches) == 1, [s.name for s in spans]
    return matches[0]


def test_tracing_is_off_by_default(settings: Settings) -> None:
    tracing.configure_tracing(settings)
    assert tracing.langchain_callbacks() == []
    with tracing.span("anything") as current:
        assert current is None


async def test_ticket_run_is_one_trace_of_nodes_llm_and_tool_spans(
    settings: Settings, exporter: in_memory.InMemorySpanExporter
) -> None:
    async with create_runtime(settings) as runtime:
        view = await runtime.service.submit(example("06").ticket)
    spans = exporter.get_finished_spans()
    root = one(spans, "ticket submit")
    assert root.parent is None
    assert root.attributes is not None
    assert root.attributes["support_agent.ticket_id"] == view.id
    assert root.attributes["support_agent.ticket.status"] == "resolved"
    assert {s.context.trace_id for s in spans if s.context} == {root.context.trace_id if root.context else 0}

    nodes = {s.name for s in children(spans, root)}
    assert nodes == {"node classify", "node research", "node lookup_tools", "node apply_policy", "node draft_reply"}

    classify = one(spans, "node classify")
    (chat,) = children(spans, classify)
    assert chat.name == "chat fake-support"
    assert chat.attributes is not None
    assert chat.attributes["gen_ai.provider.name"] == "fake"
    assert chat.attributes["gen_ai.operation.name"] == "chat"

    (tool,) = children(spans, one(spans, "node lookup_tools"))
    assert tool.name == "execute_tool search_knowledge_base"
    assert tool.attributes is not None
    assert tool.attributes["gen_ai.tool.name"] == "search_knowledge_base"
    assert tool.attributes["support_agent.kb.article_ids"][0] == "KB-001"  # type: ignore[index]
    assert all(s.status.status_code is not StatusCode.ERROR for s in spans)


async def test_interrupt_is_recorded_as_event_not_error(
    settings: Settings, exporter: in_memory.InMemorySpanExporter
) -> None:
    async with create_runtime(settings) as runtime:
        paused = await runtime.service.submit(example("02").ticket)
        await runtime.service.approve(paused.id, operator="maria")
    spans = exporter.get_finished_spans()
    approval = [s for s in spans if s.name == "node human_approval"]
    assert [e.name for e in approval[0].events] == ["interrupt"]
    assert all(s.status.status_code is not StatusCode.ERROR for s in spans)
    resume = one(spans, "ticket resume")
    assert {s.name for s in children(spans, resume)} >= {"node human_approval", "node execute_action"}
    (refund,) = children(spans, one(spans, "node execute_action"))
    assert refund.name == "execute_tool create_refund"


async def test_llm_failure_marks_spans_as_errors(settings: Settings, exporter: in_memory.InMemorySpanExporter) -> None:
    async with create_runtime(settings, llm=FailingModel()) as runtime:
        await runtime.service.submit(example("01").ticket)
    spans = exporter.get_finished_spans()
    for name in ("ticket submit", "node classify", "chat fake-support"):
        span = one(spans, name)
        assert span.status.status_code is StatusCode.ERROR, name
        assert "provider unavailable" in (span.status.description or "")


async def test_console_exporter_prints_spans(settings: Settings, capsys: pytest.CaptureFixture[str]) -> None:
    tracing.configure_tracing(settings.model_copy(update={"otel_traces_exporter": "console"}))
    try:
        async with create_runtime(settings) as runtime:
            await runtime.service.submit(example("04").ticket)
    finally:
        tracing.use_tracer(None)
    output = capsys.readouterr().out
    assert '"name": "node classify"' in output
    assert '"service.name": "support-autopilot"' in output
