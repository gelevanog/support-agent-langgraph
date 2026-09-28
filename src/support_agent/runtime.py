"""Wires settings -> store client, LLM, checkpointer, graph and service (used by API, CLI, tests)."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI
from langchain_core.language_models import BaseChatModel

from support_agent.config import Settings
from support_agent.graph import AgentDeps, SupportGraph, build_graph
from support_agent.llm import build_chat_model
from support_agent.persistence import TicketRepository, create_engine, open_checkpointer
from support_agent.service import TicketService
from support_agent.store_api import StoreRepository, create_store_app
from support_agent.tools import StoreClient, build_store_tools


@dataclass(frozen=True)
class Runtime:
    settings: Settings
    service: TicketService
    graph: SupportGraph
    store_app: FastAPI
    store: StoreRepository


@asynccontextmanager
async def create_runtime(
    settings: Settings,
    *,
    llm: BaseChatModel | None = None,
    store: StoreRepository | None = None,
) -> AsyncIterator[Runtime]:
    """Build everything the agent needs. `llm` and `store` can be injected (tests, API app)."""
    store = store or StoreRepository()
    store_app = create_store_app(store)
    client = (
        StoreClient.remote(settings.store_api_url, settings.store_api_timeout_seconds)
        if settings.store_api_url
        else StoreClient.in_process(store_app)
    )
    engine = create_engine(settings.database_url)
    repository = TicketRepository(engine)
    await repository.create_schema()
    try:
        async with client, open_checkpointer(settings.database_url) as checkpointer:
            deps = AgentDeps(
                llm=llm or build_chat_model(settings),
                tools=build_store_tools(client),
                policy=settings.policy_config(),
                store_name=settings.store_name,
                max_research_steps=settings.max_research_steps,
            )
            graph = build_graph(deps, checkpointer)
            yield Runtime(
                settings=settings,
                service=TicketService(graph, repository),
                graph=graph,
                store_app=store_app,
                store=store,
            )
    finally:
        await engine.dispose()
