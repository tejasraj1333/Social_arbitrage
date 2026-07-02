"""Unit tests for the ReconCollector base contract (no network)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sam.core.errors import CredentialsMissing
from sam.recon import collector_base as cb
from sam.recon.collector_base import ReconCollector


class _DummyCollector(ReconCollector):
    source_name = "dummy"
    required_fields = ["id", "title", "ts"]
    timestamp_field = "ts"
    sample_size = 2

    def __init__(self, records, raises=None):
        super().__init__()
        self._records = records
        self._raises = raises

    def fetch(self):
        if self._raises is not None:
            raise self._raises
        return self._records


@pytest.fixture(autouse=True)
def _isolate_data_dirs(tmp_path, monkeypatch):
    """Redirect raw/sample writes into a temp dir so tests never touch data/."""
    monkeypatch.setattr(cb, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(cb, "SAMPLE_DIR", tmp_path / "sample")


def test_validate_full_completeness():
    recs = [
        {"id": 1, "title": "a", "ts": 1700000000},
        {"id": 2, "title": "b", "ts": 1700000100},
    ]
    rep = _DummyCollector(recs).validate(recs)
    assert rep.record_count == 2
    assert rep.schema_completeness == 1.0
    assert rep.ok


def test_validate_flags_absent_field():
    recs = [{"id": 1, "title": "", "ts": None}]  # title empty + ts null -> 0% coverage
    rep = _DummyCollector(recs).validate(recs)
    assert rep.field_completeness["title"] == 0.0
    assert rep.field_completeness["ts"] == 0.0
    assert any("title" in issue for issue in rep.issues)
    assert not rep.ok


def test_save_sample_caps_and_writes():
    recs = [{"id": i, "title": str(i), "ts": 1700000000 + i} for i in range(5)]
    raw_path, sample_path = _DummyCollector(recs).save_sample(recs)
    assert raw_path is not None and sample_path is not None
    raw_lines = Path(raw_path).read_text(encoding="utf-8").splitlines()
    sample_lines = Path(sample_path).read_text(encoding="utf-8").splitlines()
    assert len(raw_lines) == 5  # full payload preserved
    assert len(sample_lines) == 2  # capped at sample_size
    assert json.loads(sample_lines[0])["id"] == 0


def test_save_sample_empty_returns_none():
    assert _DummyCollector([]).save_sample([]) == (None, None)


def test_run_ok_path():
    recs = [{"id": 1, "title": "a", "ts": 1700000000}]
    result = _DummyCollector(recs).run()
    assert result.status == "ok"
    assert result.record_count == 1
    assert result.sample_path is not None
    assert result.validation is not None


def test_run_needs_credentials():
    result = _DummyCollector([], raises=CredentialsMissing("no token")).run()
    assert result.status == "needs_credentials"
    assert result.record_count == 0
    assert result.sample_path is None
    assert "no token" in result.detail


def test_run_empty():
    result = _DummyCollector([]).run()
    assert result.status == "empty"
    assert result.record_count == 0


def test_run_unexpected_error_reports_error_status():
    """A crashing source becomes a scorecard row, not a process crash."""
    result = _DummyCollector([], raises=RuntimeError("boom")).run()
    assert result.status == "error"
    assert result.record_count == 0
    assert result.sample_path is None
    assert "RuntimeError" in result.detail and "boom" in result.detail


def test_to_epoch_treats_naive_timestamps_as_utc():
    from datetime import UTC, datetime

    expected = datetime(2026, 1, 1, tzinfo=UTC).timestamp()
    # Naive ISO string and naive datetime must both be pinned to UTC,
    # not reinterpreted in the machine's local timezone.
    assert ReconCollector._to_epoch("2026-01-01") == expected
    assert ReconCollector._to_epoch(datetime(2026, 1, 1)) == expected
    assert ReconCollector._to_epoch("2026-01-01T00:00:00Z") == expected
    assert ReconCollector._to_epoch("not-a-date") is None


def test_freshness_hours_positive_and_optional():
    recs = [{"id": 1, "title": "a", "ts": 1700000000}]
    c = _DummyCollector(recs)
    assert isinstance(c.freshness_hours(recs), float)
    c.timestamp_field = None
    assert c.freshness_hours(recs) is None


def test_estimated_monthly_volume_robust_to_outliers():
    # 100 records spaced 1h apart -> ~1/hour -> ~720/month.
    base = 1_700_000_000
    recs = [{"id": i, "title": "a", "ts": base + i * 3600} for i in range(100)]
    vol = _DummyCollector(recs).estimated_monthly_volume(recs)
    assert vol is not None and 650 <= vol <= 800

    # One ancient outlier must barely move the interquartile estimate.
    recs_outlier = [{"id": -1, "title": "a", "ts": base - 1000 * 3600}, *recs]
    vol_outlier = _DummyCollector(recs_outlier).estimated_monthly_volume(recs_outlier)
    assert vol_outlier is not None and 650 <= vol_outlier <= 800

    # Not applicable without a timestamp field / too few records.
    c = _DummyCollector(recs)
    c.timestamp_field = None
    assert c.estimated_monthly_volume(recs) is None
    assert _DummyCollector(recs[:3]).estimated_monthly_volume(recs[:3]) is None
