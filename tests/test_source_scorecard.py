"""Unit tests for the evaluation scorecard (no network; synthetic ReconResults)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from sam.evaluation import source_scorecard as sc
from sam.evaluation.source_metrics import (
    collection_difficulty,
    is_recommended,
    scorecard_row,
)
from sam.recon import collector_base as cb
from sam.recon.collector_base import ReconResult


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "SAMPLE_DIR", tmp_path / "sample")
    monkeypatch.setattr(sc, "DOCS_DIR", tmp_path / "docs")


def _result(name, status="ok", n=100, comp=1.0, fresh=1.0, vol=1000):
    return ReconResult(
        source_name=name,
        status=status,
        record_count=n,
        schema_completeness=comp,
        freshness_hours=fresh,
        sample_path=f"data/sample/{name}.jsonl",
        raw_path=f"data/raw/{name}.jsonl",
        estimated_monthly_volume=vol,
    )


def _needs_creds(name):
    return _result(name, status="needs_credentials", n=0, comp=0.0, fresh=None, vol=None)


def test_recommendation_rule():
    assert is_recommended(_result("rss"))
    assert not is_recommended(_result("x", comp=0.5))  # below completeness floor
    assert not is_recommended(_needs_creds("reddit"))  # no data yet


def test_scorecard_row_includes_difficulty():
    row = scorecard_row(_result("rss"))
    assert row["collection_difficulty"] == collection_difficulty("rss")
    assert set(row) == set(sc.columns())


def test_build_dataframe_columns_and_order():
    results = [_result("rss"), _result("yahoo", fresh=None), _needs_creds("reddit")]
    df = sc.build_dataframe(results)
    assert list(df.columns) == sc.columns()
    assert df["source"].tolist() == ["rss", "yahoo", "reddit"]


def test_write_outputs_creates_csv_and_md():
    df = sc.build_dataframe([_result("rss"), _needs_creds("kaggle")])
    csv, md = sc.write_outputs(df)
    assert Path(csv).exists() and Path(md).exists()
    reloaded = pd.read_csv(csv)
    assert reloaded["source"].tolist() == ["rss", "kaggle"]
    text = Path(md).read_text(encoding="utf-8")
    assert "# Source Scorecard" in text
    assert "| source |" in text
    assert "needs_credentials" in text


def test_generate_does_not_run_collectors_when_results_supplied(monkeypatch):
    def _boom():
        raise AssertionError("run_all() must not be called when results are supplied")

    monkeypatch.setattr(sc, "run_all", _boom)
    df = sc.generate(results=[_result("rss")])
    assert len(df) == 1
