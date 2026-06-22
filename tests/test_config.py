"""Config layering: YAML defaults + env override + nested delimiter."""

from __future__ import annotations

import pytest

from sam.core.config import Settings


def test_defaults() -> None:
    s = Settings()
    assert s.db.port == 5432
    assert s.db.url.startswith("postgresql+psycopg://")


def test_env_nested_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SAM_DB__HOST", "db.internal")
    monkeypatch.setenv("SAM_DB__PORT", "6543")
    s = Settings()
    assert s.db.host == "db.internal"
    assert s.db.port == 6543
    assert "db.internal:6543" in s.db.url
