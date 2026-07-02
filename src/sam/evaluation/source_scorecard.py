"""Source scorecard: run every recon collector and tabulate the results.

Produces two artifacts:
  - data/sample/source_scorecard.csv  (committed machine-readable evidence)
  - docs/source_scorecard.md          (human-readable table + legend)

Credentialed sources that lack creds appear with status="needs_credentials"
(record_count 0) — documented, not hidden.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from sam.collectors.hn_collector import HackerNewsCollector
from sam.collectors.kaggle_evaluator import KaggleEvaluator
from sam.collectors.reddit_collector import RedditCollector
from sam.collectors.rss_collector import RSSCollector
from sam.collectors.yahoo_collector import YahooCollector
from sam.core.config import PROJECT_ROOT
from sam.core.logging import get_logger
from sam.evaluation.source_metrics import (
    columns,
    difficulty_note,
    scorecard_row,
)
from sam.recon import collector_base as cb
from sam.recon.collector_base import ReconCollector, ReconResult

log = get_logger("evaluation.scorecard")

COLLECTORS: list[type[ReconCollector]] = [
    RSSCollector,
    YahooCollector,
    HackerNewsCollector,
    RedditCollector,
    KaggleEvaluator,
]

DOCS_DIR = PROJECT_ROOT / "docs"


def run_all() -> list[ReconResult]:
    """Run every collector once (network) and return their ReconResults."""
    return [cls().run() for cls in COLLECTORS]


def build_dataframe(results: list[ReconResult]) -> pd.DataFrame:
    rows = [scorecard_row(r) for r in results]
    return pd.DataFrame(rows, columns=columns())


def csv_path() -> Path:
    return cb.SAMPLE_DIR / "source_scorecard.csv"


def _df_to_markdown(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        cells = ["" if pd.isna(v) else str(v) for v in row.tolist()]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _render_md(df: pd.DataFrame) -> str:
    ts = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "# Source Scorecard",
        "",
        f"_Generated {ts} by `sam recon`. Phase-1 source reconnaissance._",
        "",
        _df_to_markdown(df),
        "",
        "## Legend",
        "",
        "- **schema_completeness** — mean fraction of required fields present "
        "(non-null) across fetched records (1.0 = every required field on every record).",
        "- **freshness_hours** — age of the most recent record at collection time "
        "(blank where not time-stamped, e.g. price bars / dataset metadata).",
        "- **estimated_monthly_volume** — records/month extrapolated from the sample's "
        "observed time span (order-of-magnitude; top-N feeds compress the span).",
        "- **collection_difficulty** — operational effort/risk to run at scale.",
        "- **recommended** — `True` when status is `ok` and completeness ≥ 0.80.",
        "- **needs_credentials** — collector is built + unit-tested; runs live once "
        "credentials are supplied (see `docs/legal_register.md`).",
        "",
        "## Collection-difficulty notes",
        "",
    ]
    for source in df["source"].tolist():
        parts.append(f"- **{source}** — {difficulty_note(source)}")
    return "\n".join(parts) + "\n"


def write_outputs(df: pd.DataFrame) -> tuple[Path, Path]:
    cb.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    csv = csv_path()
    md = DOCS_DIR / "source_scorecard.md"
    df.to_csv(csv, index=False)
    md.write_text(_render_md(df), encoding="utf-8")
    return csv, md


def generate(results: list[ReconResult] | None = None) -> pd.DataFrame:
    """Build + persist the scorecard. Runs all collectors if results not supplied."""
    if results is None:
        results = run_all()
    df = build_dataframe(results)
    csv, md = write_outputs(df)
    log.info("scorecard_written", csv=str(csv), md=str(md), sources=len(df))
    return df
