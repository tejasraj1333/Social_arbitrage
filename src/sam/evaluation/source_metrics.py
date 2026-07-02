"""Per-source validation metrics for the recon scorecard.

Most numeric metrics (record_count, schema_completeness, freshness_hours,
estimated_monthly_volume) are computed by `ReconCollector.run()` and carried on
`ReconResult`. This module adds the qualitative judgements — collection
difficulty and a production-recommendation rule — and assembles a flat scorecard
row from a `ReconResult`.
"""

from __future__ import annotations

from typing import Any, Literal

from sam.recon.collector_base import ReconResult

DifficultyRating = Literal["low", "medium", "high", "unknown"]

# How hard each source is to collect at production scale, with the rationale.
_DIFFICULTY: dict[str, DifficultyRating] = {
    "rss": "low",
    "yahoo": "low",
    "hackernews": "low",
    "reddit": "medium",
    "kaggle": "medium",
}
_DIFFICULTY_NOTE: dict[str, str] = {
    "rss": "public feeds, no auth; only flakiness is dead/changed feed URLs",
    "yahoo": "unofficial API, no auth; risk is undocumented rate limits / breakage",
    "hackernews": "official keyless API; 1 request per item is the only friction",
    "reddit": "OAuth app required; 100 req/min and listing-depth caps",
    "kaggle": "API token required; redistribution constrained by per-dataset license",
}

# Completeness floor for a source to be considered production-ready.
_COMPLETENESS_FLOOR = 0.80

_COLUMNS = [
    "source",
    "status",
    "record_count",
    "schema_completeness",
    "freshness_hours",
    "estimated_monthly_volume",
    "collection_difficulty",
    "recommended",
]


def collection_difficulty(source_name: str) -> DifficultyRating:
    return _DIFFICULTY.get(source_name, "unknown")


def difficulty_note(source_name: str) -> str:
    return _DIFFICULTY_NOTE.get(source_name, "")


def is_recommended(result: ReconResult) -> bool:
    """Recommend a source only if it fetched usable data at high completeness."""
    return (
        result.status == "ok"
        and result.record_count > 0
        and result.schema_completeness >= _COMPLETENESS_FLOOR
    )


def scorecard_row(result: ReconResult) -> dict[str, Any]:
    """Flatten a ReconResult into a single scorecard row."""
    return {
        "source": result.source_name,
        "status": result.status,
        "record_count": result.record_count,
        "schema_completeness": result.schema_completeness,
        "freshness_hours": result.freshness_hours,
        "estimated_monthly_volume": result.estimated_monthly_volume,
        "collection_difficulty": collection_difficulty(result.source_name),
        "recommended": is_recommended(result),
    }


def columns() -> list[str]:
    return list(_COLUMNS)
