"""Bronze lake writer tests (tmp dir; no real data/ touched)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sam.ingestion import lake


@pytest.fixture(autouse=True)
def _isolate_lake(tmp_path, monkeypatch):
    monkeypatch.setattr(lake, "RAW_LAKE_DIR", tmp_path / "00_raw")


RECORDS = [
    {"title": "A", "url": "https://x/a", "score": 1},
    {"title": "Bücher", "url": "https://x/b", "score": 2},  # non-ASCII round-trip
]


def test_write_creates_partitioned_gzip_and_manifest() -> None:
    artifact = lake.write_raw("rss", RECORDS)
    assert artifact is not None
    data_path = Path(artifact.path)
    assert data_path.exists()
    assert data_path.suffix == ".gz"
    assert "dt=" in data_path.parent.name  # ingestion-date partition
    assert data_path.parent.parent.name == "rss"

    manifest = json.loads(Path(artifact.manifest_path).read_text(encoding="utf-8"))
    assert manifest["rows"] == 2
    assert manifest["sha256"] == artifact.sha256
    assert manifest["path"] == artifact.path


def test_round_trip_preserves_records() -> None:
    artifact = lake.write_raw("rss", RECORDS)
    assert artifact is not None
    assert lake.read_raw(artifact.path) == RECORDS


def test_append_only_never_overwrites() -> None:
    first = lake.write_raw("rss", RECORDS)
    second = lake.write_raw("rss", RECORDS)
    assert first is not None and second is not None
    assert first.path != second.path  # unique nonce per write
    assert Path(first.path).exists() and Path(second.path).exists()


def test_empty_batch_writes_nothing() -> None:
    assert lake.write_raw("rss", []) is None
    assert not (lake.RAW_LAKE_DIR / "rss").exists()
