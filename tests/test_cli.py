"""CLI smoke test."""

from __future__ import annotations

import pytest

from sam.cli import main


def test_cli_check_returns_zero() -> None:
    assert main(["check"]) == 0


def test_cli_recon_single_source(monkeypatch) -> None:
    # Stub the collector run so the CLI path is exercised without any network.
    from sam.collectors import rss_collector as rc
    from sam.recon.collector_base import ReconResult

    def fake_run(self):
        return ReconResult(
            source_name="rss",
            status="ok",
            record_count=100,
            schema_completeness=1.0,
            freshness_hours=0.1,
            sample_path="data/sample/rss.jsonl",
            raw_path="data/raw/rss.jsonl",
            estimated_monthly_volume=60,
        )

    monkeypatch.setattr(rc.RSSCollector, "run", fake_run)
    assert main(["recon", "--source", "rss"]) == 0


def test_cli_recon_error_status_exits_nonzero(monkeypatch) -> None:
    from sam.collectors import rss_collector as rc
    from sam.recon.collector_base import ReconResult

    def fake_run(self):
        return ReconResult(
            source_name="rss",
            status="error",
            record_count=0,
            schema_completeness=0.0,
            freshness_hours=None,
            sample_path=None,
            raw_path=None,
            detail="ConnectError: boom",
        )

    monkeypatch.setattr(rc.RSSCollector, "run", fake_run)
    assert main(["recon", "--source", "rss"]) == 1


def test_cli_recon_rejects_unknown_source() -> None:
    with pytest.raises(SystemExit):  # argparse choices validation
        main(["recon", "--source", "twitter"])
