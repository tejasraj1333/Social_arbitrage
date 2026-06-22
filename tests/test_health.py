"""Liveness probe must work without any external dependency."""

from __future__ import annotations

from fastapi.testclient import TestClient

from sam import __version__


def test_health_ok(client: TestClient) -> None:
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["version"] == __version__


def test_ready_reports_db_state(client: TestClient) -> None:
    # No DB in unit tests -> readiness should degrade gracefully, not 500.
    resp = client.get("/ready")
    assert resp.status_code == 200
    assert resp.json()["database"] in {"ok", "unreachable"}
