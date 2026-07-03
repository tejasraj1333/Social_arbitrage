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


# ---------------------------------------------------------------- sam ingest


def _ingest_result(status: str = "success"):
    from sam.ingestion.runner import IngestResult

    return IngestResult(
        source_name="rss",
        status=status,
        rows_fetched=10,
        rows_inserted=4,
        raw_path="data/00_raw/rss/dt=2026-07-02/x.jsonl.gz",
        run_id=1,
        detail="" if status == "success" else "RuntimeError: boom",
    )


def test_cli_ingest_single_source_success(monkeypatch) -> None:
    from sam.ingestion import runner as runner_mod

    monkeypatch.setattr(
        runner_mod.IngestionRunner, "run_many", lambda self, names, backfill: [_ingest_result()]
    )
    assert main(["ingest", "--source", "rss"]) == 0


def test_cli_ingest_error_exits_nonzero(monkeypatch) -> None:
    from sam.ingestion import runner as runner_mod

    monkeypatch.setattr(
        runner_mod.IngestionRunner,
        "run_many",
        lambda self, names, backfill: [_ingest_result("error")],
    )
    assert main(["ingest", "--source", "rss"]) == 1


def test_cli_ingest_all_expands_registry(monkeypatch) -> None:
    from sam.ingestion import runner as runner_mod

    seen: dict[str, list[str]] = {}

    def fake_run_many(self, names, backfill):
        seen["names"] = list(names)
        return [_ingest_result()]

    monkeypatch.setattr(runner_mod.IngestionRunner, "run_many", fake_run_many)
    assert main(["ingest"]) == 0
    assert seen["names"] == ["rss", "yahoo", "hackernews"]


def test_cli_ingest_loop_cycles_then_exits(monkeypatch) -> None:
    from sam.cli import _run_ingest
    from sam.ingestion import runner as runner_mod

    calls = {"run": 0, "sleep": 0}

    def fake_run_many(self, names, backfill):
        calls["run"] += 1
        return [_ingest_result()]

    monkeypatch.setattr(runner_mod.IngestionRunner, "run_many", fake_run_many)
    monkeypatch.setattr(
        "sam.cli.time.sleep", lambda s: calls.__setitem__("sleep", calls["sleep"] + 1)
    )

    assert _run_ingest("rss", loop_seconds=1, max_cycles=3) == 0
    assert calls["run"] == 3
    assert calls["sleep"] == 2  # no sleep after the final cycle


def test_cli_seed_uses_config_universe(monkeypatch, db_session) -> None:
    from sqlalchemy import select

    from sam.ingestion import runner as runner_mod
    from sam.storage.models import Entity

    monkeypatch.setattr(runner_mod, "default_session", lambda: db_session)
    assert main(["seed"]) == 0
    tickers = set(db_session.execute(select(Entity.ticker)).scalars())
    assert {"AAPL", "MSFT", "NVDA", "TSLA", "AMZN", "AMD"} == tickers


def test_cli_seed_update_syncs_aliases_from_config(monkeypatch, db_session) -> None:
    from sqlalchemy import select

    from sam.ingestion import runner as runner_mod
    from sam.storage.models import Entity

    monkeypatch.setattr(runner_mod, "default_session", lambda: db_session)
    assert main(["seed"]) == 0

    # Wipe one row's aliases to simulate a pre-P3 database, then --update.
    nvda = db_session.execute(select(Entity).where(Entity.ticker == "NVDA")).scalar_one()
    nvda.aliases = []
    db_session.commit()

    assert main(["seed", "--update"]) == 0
    refreshed = db_session.execute(select(Entity).where(Entity.ticker == "NVDA")).scalar_one()
    assert refreshed.aliases == ["Nvidia"]  # curated value from config/sources.yaml


def test_cli_resolve_links_entities(monkeypatch, db_session) -> None:
    from sqlalchemy import select

    from sam.ingestion import runner as runner_mod
    from sam.processing import pipeline as pipeline_mod
    from sam.storage.models import DocumentEntity
    from sam.storage.repositories import DocumentRepository, SourceRepository

    monkeypatch.setattr(runner_mod, "default_session", lambda: db_session)
    assert main(["seed"]) == 0

    SourceRepository(db_session).get_or_create("rss", "rss")
    DocumentRepository(db_session).upsert_many(
        [
            {
                "source_id": 1,
                "external_id": "x",
                "url": "https://example.com/x",
                "author": None,
                "title": "Nvidia and $TSLA both rallied",
                "raw_text": None,
                "lang": None,
                "content_hash": "e" * 64,
                "published_at": None,
                "engagement": {},
            }
        ]
    )
    db_session.commit()

    # Pipeline resolves its factory lazily, so patching the module works.
    monkeypatch.setattr(pipeline_mod, "default_session", lambda: db_session)
    assert main(["resolve"]) == 0
    links = db_session.execute(select(DocumentEntity)).scalars().all()
    assert {link.method for link in links} == {"alias", "cashtag"}


def test_cli_resolve_evaluate_passes_gate() -> None:
    # Runs the real labeled sample against the config universe — no DB, no
    # network. Exit 0 == precision gate met.
    assert main(["resolve", "--evaluate"]) == 0


def test_cli_dq_prints_table_and_persists(monkeypatch, db_session, capsys) -> None:
    from sqlalchemy import select

    from sam.processing import quality as quality_mod
    from sam.storage.models import DataQualityCheck
    from sam.storage.repositories import IngestionRunRepository, SourceRepository

    source = SourceRepository(db_session).get_or_create("rss", "rss")
    repo = IngestionRunRepository(db_session)
    repo.finish(repo.start(source.id), status="success", rows_fetched=12)
    db_session.commit()

    monkeypatch.setattr(quality_mod, "default_session", lambda: db_session)
    assert main(["dq"]) == 0
    out = capsys.readouterr().out
    assert "duplicate_rate" in out and "freshness" in out
    assert db_session.execute(select(DataQualityCheck)).scalars().all()


def test_cli_runs_prints_table(monkeypatch, db_session, capsys) -> None:
    from sam.ingestion import runner as runner_mod
    from sam.storage.repositories import IngestionRunRepository, SourceRepository

    source = SourceRepository(db_session).get_or_create("rss", "rss")
    repo = IngestionRunRepository(db_session)
    repo.finish(repo.start(source.id), status="success", rows_fetched=10, rows_inserted=3)
    db_session.commit()

    monkeypatch.setattr(runner_mod, "default_session", lambda: db_session)
    assert main(["runs", "--limit", "5"]) == 0
    out = capsys.readouterr().out
    assert "rss" in out and "success" in out
