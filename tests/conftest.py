"""Shared fixtures. M0 keeps tests DB-free so they run anywhere (incl. CI)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from sam.api.app import create_app


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())
