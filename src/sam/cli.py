"""Minimal CLI entrypoint (`sam ...`).

Expands per milestone. Currently exposes:
  sam --version
  sam check                       validate configuration and exit
  sam recon [--source SOURCE]     run Phase-1 source-recon collector(s)
  sam seed                        seed entities from config/sources.yaml
  sam ingest [--source SOURCE] [--backfill] [--loop SECONDS]
                                  run production ingestion (Phase 2)
  sam runs [--limit N]            show recent ingestion runs
"""

from __future__ import annotations

import argparse
import time

from sam import __version__
from sam.core.config import get_settings
from sam.core.logging import configure_logging, get_logger

_RECON_SOURCES = ["all", "rss", "yahoo", "hackernews", "reddit", "kaggle"]
# Production ingestion sources (kaggle/reddit join once credentialed/built).
_INGEST_SOURCES = ["all", "rss", "yahoo", "hackernews"]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sam", description="Social Arbitrage Model CLI")
    parser.add_argument("--version", action="version", version=f"sam {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("check", help="Validate configuration and exit")

    recon = sub.add_parser("recon", help="Run Phase-1 source-recon collector(s)")
    recon.add_argument(
        "--source",
        default="all",
        choices=_RECON_SOURCES,
        help="Source to recon; 'all' (default) runs every collector and writes the scorecard.",
    )

    sub.add_parser("seed", help="Seed the entities table from config/sources.yaml")

    ingest = sub.add_parser("ingest", help="Run production ingestion collector(s)")
    ingest.add_argument(
        "--source",
        default="all",
        choices=_INGEST_SOURCES,
        help="Source to ingest; 'all' (default) runs every production collector.",
    )
    ingest.add_argument(
        "--backfill",
        action="store_true",
        help="Fetch the full historical window (e.g. Yahoo 1y) instead of the incremental one.",
    )
    ingest.add_argument(
        "--loop",
        type=int,
        default=None,
        metavar="SECONDS",
        help="Minimal scheduler: re-run every N seconds until interrupted "
        "(prefer cron/Task Scheduler in production).",
    )

    runs = sub.add_parser("runs", help="Show recent ingestion runs (observability)")
    runs.add_argument("--limit", type=int, default=20, help="How many runs to show.")

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    log = get_logger("sam.cli")

    if args.command == "check":
        log.info("config.ok", env=settings.env, db_host=settings.db.host)
        return 0

    if args.command == "recon":
        return _run_recon(args.source)

    if args.command == "seed":
        return _run_seed()

    if args.command == "ingest":
        return _run_ingest(args.source, backfill=args.backfill, loop_seconds=args.loop)

    if args.command == "runs":
        return _show_runs(args.limit)

    parser.print_help()
    return 0


def _run_recon(source: str) -> int:
    """Run collector(s); exit non-zero when any source errored (scheduler-friendly)."""
    # Lazy imports: keep `sam check` / `--version` free of heavy deps (pandas, yfinance).
    from sam.collectors.hn_collector import HackerNewsCollector
    from sam.collectors.kaggle_evaluator import KaggleEvaluator
    from sam.collectors.reddit_collector import RedditCollector
    from sam.collectors.rss_collector import RSSCollector
    from sam.collectors.yahoo_collector import YahooCollector
    from sam.evaluation.source_scorecard import generate

    registry = {
        "rss": RSSCollector,
        "yahoo": YahooCollector,
        "hackernews": HackerNewsCollector,
        "reddit": RedditCollector,
        "kaggle": KaggleEvaluator,
    }
    logger = get_logger("sam.recon")

    if source == "all":
        df = generate()  # runs every collector + writes CSV/MD scorecard
        for row in df.to_dict("records"):
            logger.info(
                "recon.source",
                source=row["source"],
                status=row["status"],
                records=row["record_count"],
                completeness=row["schema_completeness"],
                recommended=row["recommended"],
            )
        errored = df.loc[df["status"] == "error", "source"].tolist()
        logger.info("recon.complete", sources=len(df), errored=errored)
        return 1 if errored else 0

    result = registry[source]().run()
    logger.info(
        "recon.complete",
        source=result.source_name,
        status=result.status,
        records=result.record_count,
        sample=result.sample_path,
    )
    return 1 if result.status == "error" else 0


def _run_seed() -> int:
    """Seed the entities table from the configured ticker universe."""
    from sqlalchemy.exc import SQLAlchemyError

    from sam.ingestion import runner as runner_mod
    from sam.recon.sources import load_sources
    from sam.storage.repositories import EntityRepository

    logger = get_logger("sam.seed")
    universe = load_sources().get("universe", [])
    try:
        session = runner_mod.default_session()
        try:
            inserted = EntityRepository(session).seed(universe)
            session.commit()
        finally:
            session.close()
    except SQLAlchemyError as exc:
        logger.error(
            "seed_db_error",
            error=str(exc),
            hint="is Postgres up? (docker compose up -d db; SAM_DB__PORT=5433 for compose)",
        )
        return 1
    logger.info("seed_done", requested=len(universe), inserted=inserted)
    return 0


def _run_ingest(
    source: str,
    *,
    backfill: bool = False,
    loop_seconds: int | None = None,
    max_cycles: int | None = None,
) -> int:
    """Run ingestion once, or repeatedly with --loop (minimal scheduler).

    Exit code 1 when any source errored in the (final) cycle. `max_cycles`
    exists for tests; the CLI loop runs until interrupted.
    """
    from sqlalchemy.exc import SQLAlchemyError

    from sam.ingestion import runner as runner_mod

    logger = get_logger("sam.ingest")
    names = list(runner_mod.SOURCES) if source == "all" else [source]
    runner = runner_mod.IngestionRunner()

    cycles = 0
    while True:
        cycles += 1
        try:
            results = runner.run_many(names, backfill=backfill)
            errored = [r.source_name for r in results if r.status == "error"]
            for r in results:
                logger.info(
                    "ingest.source",
                    source=r.source_name,
                    status=r.status,
                    fetched=r.rows_fetched,
                    inserted=r.rows_inserted,
                    run_id=r.run_id,
                )
            logger.info("ingest.cycle_complete", cycle=cycles, errored=errored)
        except SQLAlchemyError as exc:
            errored = names
            logger.error(
                "ingest_db_error",
                error=str(exc),
                hint="is Postgres up? (docker compose up -d db; SAM_DB__PORT=5433 for compose)",
            )
        if loop_seconds is None:
            return 1 if errored else 0
        if max_cycles is not None and cycles >= max_cycles:
            return 1 if errored else 0
        time.sleep(loop_seconds)


def _show_runs(limit: int) -> int:
    """Print recent ingestion runs as a compact table (observability)."""
    from sqlalchemy import select
    from sqlalchemy.exc import SQLAlchemyError

    from sam.ingestion import runner as runner_mod
    from sam.storage.models import Source
    from sam.storage.repositories import IngestionRunRepository

    logger = get_logger("sam.runs")
    try:
        session = runner_mod.default_session()
        try:
            names = dict(session.execute(select(Source.id, Source.name)).tuples().all())
            recent = IngestionRunRepository(session).recent(limit=limit)
            header = (
                f"{'id':>5}  {'source':<12} {'status':<8} {'started (UTC)':<20} "
                f"{'fetched':>8} {'inserted':>9}  error"
            )
            print(header)
            print("-" * len(header))
            for run in recent:
                started = run.started_at.strftime("%Y-%m-%d %H:%M:%S") if run.started_at else "-"
                error = (run.error or "")[:60]
                print(
                    f"{run.id:>5}  {names.get(run.source_id, '?'):<12} {run.status:<8} "
                    f"{started:<20} {run.rows_fetched:>8} {run.rows_inserted:>9}  {error}"
                )
        finally:
            session.close()
    except SQLAlchemyError as exc:
        logger.error("runs_db_error", error=str(exc))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
