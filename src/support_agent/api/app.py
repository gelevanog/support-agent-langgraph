"""FastAPI application: REST API, operator UI and the mounted mock Store API."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, RedirectResponse
from langchain_core.language_models import BaseChatModel

from support_agent import __version__
from support_agent.api.routes import router
from support_agent.config import Settings, get_settings
from support_agent.logging_config import configure_logging
from support_agent.runtime import create_runtime
from support_agent.service import InvalidTicketStateError, TicketNotFoundError
from support_agent.store_api import StoreRepository, create_store_app
from support_agent.ui.routes import mount_ui


def create_app(settings: Settings | None = None, *, llm: BaseChatModel | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, settings.log_format)
    store = StoreRepository()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        async with create_runtime(settings, llm=llm, store=store) as runtime:
            app.state.runtime = runtime
            yield

    app = FastAPI(
        title="Support Autopilot",
        description=(
            "LangGraph agent that resolves e-commerce support tickets with deterministic business "
            "rules and human approval."
        ),
        version=__version__,
        lifespan=lifespan,
    )
    app.state.settings = settings

    @app.exception_handler(TicketNotFoundError)
    async def _not_found(_: Request, exc: TicketNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_404_NOT_FOUND, content={"detail": f"Ticket {exc} not found"})

    @app.exception_handler(InvalidTicketStateError)
    async def _conflict(_: Request, exc: InvalidTicketStateError) -> JSONResponse:
        return JSONResponse(status_code=status.HTTP_409_CONFLICT, content={"detail": str(exc)})

    @app.get("/", include_in_schema=False)
    async def root() -> RedirectResponse:
        return RedirectResponse(url="/ui")

    app.include_router(router)
    mount_ui(app)
    # The mock backend is also reachable over HTTP (e.g. GET /store/orders/1042) for exploration.
    app.mount("/store", create_store_app(store))
    return app
