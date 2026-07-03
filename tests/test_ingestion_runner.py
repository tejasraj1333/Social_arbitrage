"""Runner orchestration tests: fake collectors, real (SQLite) bookkeeping."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from sam.ingestion import lake
from sam.ingestion import runner as runner_mod
from sam.ingestion.base import Collector
from sam.ingestion.runner import IngestionRunner, SourceSpec
from sam.storage.models import IngestionRun, Source


class _FakeCollector(Collector):
    source_type = "fake"
    source_name = "fake"

    def __init__(self, raw: list[dict[str, Any]], fail: str | None = None) -> None:
        self._raw = raw
        self._fail = fail

    def fetch(self) -> Iterable[dict[str, Any]]:
        if self._fail == "fetch":
            raise RuntimeError("upstream down")
        return list(self._raw)

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        return raw

    def persist(self, documents: Iterable[dict[str, Any]]) -> int:
        if self._fail == "persist":
            raise RuntimeError("db exploded")
        return len(list(documents))


def _spec(name: str, raw: list[dict[str, Any]], fail: str | None = None) -> SourceSpec:
    return SourceSpec(
        name=name,
        type="fake",
        config_ref=None,
        factory=lambda session, sid, backfill: _FakeCollector(raw, fail),
    )


@pytest.fixture(autouse=True)
def _isolate_lake(tmp_path, monkeypatch):
    monkeypatch.setattr(lake, "RAW_LAKE_DIR", tmp_path / "00_raw")


@pytest.fixture
def runner(db_session: Session) -> IngestionRunner:
    return IngestionRunner(session_factory=lambda: db_session)


def test_success_records_run_and_lake(
    runner: IngestionRunner, db_session: Session, monkeypatch
) -> None:
    raw = [{"a": 1}, {"a": 2}, {"a": 3}]
    monkeypatch.setitem(runner_mod.SOURCES, "fake", _spec("fake", raw))

    result = runner.run("fake")

    assert result.status == "success"
    assert result.rows_fetched == 3
    assert result.rows_inserted == 3
    assert result.raw_path is not None and Path(result.raw_path).exists()
    assert lake.read_raw(result.raw_path) == raw

    run_row = db_session.execute(select(IngestionRun)).scalar_one()
    assert run_row.status == "success"
    assert run_row.rows_fetched == 3
    assert run_row.rows_inserted == 3
    assert run_row.raw_path == result.raw_path
    assert run_row.finished_at is not None

    source = db_session.execute(select(Source)).scalar_one()
    assert source.name == "fake"


def test_fetch_failure_records_error_run(
    runner: IngestionRunner, db_session: Session, monkeypatch
) -> None:
    monkeypatch.setitem(runner_mod.SOURCES, "fake", _spec("fake", [], fail="fetch"))

    result = runner.run("fake")

    assert result.status == "error"
    assert "RuntimeError" in result.detail and "upstream down" in result.detail
    run_row = db_session.execute(select(IngestionRun)).scalar_one()
    assert run_row.status == "error"
    assert run_row.error is not None and "upstream down" in run_row.error
    assert run_row.raw_path is None  # nothing fetched -> nothing in the lake


def test_persist_failure_keeps_lake_artifact(
    runner: IngestionRunner, db_session: Session, monkeypatch
) -> None:
    raw = [{"a": 1}]
    monkeypatch.setitem(runner_mod.SOURCES, "fake", _spec("fake", raw, fail="persist"))

    result = runner.run("fake")

    assert result.status == "error"
    assert result.rows_fetched == 1
    # Raw is preserved for reprocessing even though persistence failed.
    assert result.raw_path is not None and Path(result.raw_path).exists()
    run_row = db_session.execute(select(IngestionRun)).scalar_one()
    assert run_row.status == "error"
    assert run_row.raw_path == result.raw_path


def test_run_many_isolates_failures(
    runner: IngestionRunner, db_session: Session, monkeypatch
) -> None:
    monkeypatch.setitem(runner_mod.SOURCES, "ok", _spec("ok", [{"a": 1}]))
    monkeypatch.setitem(runner_mod.SOURCES, "bad", _spec("bad", [], fail="fetch"))

    results = runner.run_many(["ok", "bad"])

    assert [r.status for r in results] == ["success", "error"]
    rows = db_session.execute(select(IngestionRun)).scalars().all()
    assert len(rows) == 2  # both runs recorded


def test_reusing_source_does_not_duplicate_source_rows(
    runner: IngestionRunner, db_session: Session, monkeypatch
) -> None:
    monkeypatch.setitem(runner_mod.SOURCES, "fake", _spec("fake", [{"a": 1}]))
    runner.run("fake")
    runner.run("fake")
    assert len(db_session.execute(select(Source)).scalars().all()) == 1
    assert len(db_session.execute(select(IngestionRun)).scalars().all()) == 2
