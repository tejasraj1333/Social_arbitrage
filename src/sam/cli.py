"""Minimal CLI entrypoint (`sam ...`).

Expands per milestone (e.g. `sam run pipeline`, `sam ingest reddit`).
For M0 it exposes version + a config sanity check.
"""

from __future__ import annotations

import argparse

from sam import __version__
from sam.core.config import get_settings
from sam.core.logging import configure_logging, get_logger


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sam", description="Social Arbitrage Model CLI")
    parser.add_argument("--version", action="version", version=f"sam {__version__}")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("check", help="Validate configuration and exit")

    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(level=settings.log_level, json_output=settings.log_json)
    log = get_logger("sam.cli")

    if args.command == "check":
        log.info("config.ok", env=settings.env, db_host=settings.db.host)
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
