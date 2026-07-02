"""Minimal CLI entrypoint (`sam ...`).

Expands per milestone. Currently exposes:
  sam --version
  sam check                       validate configuration and exit
  sam recon [--source SOURCE]     run Phase-1 source-recon collector(s)
"""

from __future__ import annotations

import argparse

from sam import __version__
from sam.core.config import get_settings
from sam.core.logging import configure_logging, get_logger

_RECON_SOURCES = ["all", "rss", "yahoo", "hackernews", "reddit", "kaggle"]


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

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    log = get_logger("sam.cli")

    if args.command == "check":
        log.info("config.ok", env=settings.env, db_host=settings.db.host)
        return 0

    if args.command == "recon":
        return _run_recon(args.source)

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


if __name__ == "__main__":
    raise SystemExit(main())
