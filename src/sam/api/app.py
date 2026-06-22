"""FastAPI application factory.

Keep this thin: wire config/logging, register routers and exception handlers.
Business logic lives in feature modules, not here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from sam import __version__
from sam.api.routers import health
from sam.core.config import get_settings
from sam.core.errors import NotFoundError, SAMError, ValidationError
from sam.core.logging import configure_logging, get_logger

log = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    log.info("api.startup", env=settings.env, version=__version__)
    yield
    log.info("api.shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="Social Arbitrage Model",
        version=__version__,
        summary="Detect attention & sentiment shifts before earnings/price.",
        lifespan=lifespan,
    )

    app.include_router(health.router)

    @app.exception_handler(NotFoundError)
    async def _not_found(_: Request, exc: NotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(ValidationError)
    async def _bad_request(_: Request, exc: ValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": str(exc)})

    @app.exception_handler(SAMError)
    async def _internal(_: Request, exc: SAMError) -> JSONResponse:
        log.error("api.unhandled_domain_error", error=str(exc))
        return JSONResponse(status_code=500, content={"detail": "Internal error"})

    return app


app = create_app()
