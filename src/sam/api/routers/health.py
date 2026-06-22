"""Liveness & readiness probes.

- /health : process is up (no dependencies checked) — used by load balancers.
- /ready  : dependencies reachable (DB) — used by orchestrators before routing.
"""

from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel
from sqlalchemy import text

from sam import __version__
from sam.core.db import get_engine
from sam.core.logging import get_logger

router = APIRouter(tags=["health"])
log = get_logger(__name__)


class HealthResponse(BaseModel):
    status: str
    version: str


class ReadyResponse(BaseModel):
    status: str
    database: str


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", version=__version__)


@router.get("/ready", response_model=ReadyResponse)
def ready() -> ReadyResponse:
    db_status = "ok"
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:  # noqa: BLE001 - report, don't crash the probe
        log.warning("readiness.db_unreachable", error=str(exc))
        db_status = "unreachable"
    status = "ok" if db_status == "ok" else "degraded"
    return ReadyResponse(status=status, database=db_status)
