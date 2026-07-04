"""Data-quality framework (Phase 3): checks that write auditable rows.

Every check run appends pass/warn/fail rows to ``data_quality_checks`` so
quality is a monitored time series, not a one-off script. Principles from
the blueprint: quarantine don't delete (offending ids go into ``details``),
weight don't filter, and alert on *staleness*, not just errors — a silently
stalled collector degrades every downstream signal.

Checks:

  duplicate_rate       near-duplicate share of the recent document window
                       (token-Jaccard on titles; the P3 "<2% dup rate" gate).
                       Exact duplicates cannot exist (content_hash UNIQUE);
                       what this catches is cross-feed syndication.
  freshness            hours since each source's last successful run.
  volume_anomaly       latest fetch size vs the source's trailing mean.
  resolution_coverage  share of resolved documents holding >=1 entity link
                       (with the unresolved backlog in details).
  enrichment_coverage  share of documents enriched by the NLP pipeline
                       (Phase 4; warns when a sizable corpus has zero
                       sentiment rows — pipeline rot).
  sai_freshness        days the SAI panel trails its expected last closed
                       day (Phase 5; a stalled panel starves the M5
                       validation exactly like a stalled collector).

Bot/spam scoring (blueprint W5) needs author-level Reddit data and joins
this module once Reddit credentials land.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from sam.core.db import default_session
from sam.core.logging import get_logger
from sam.storage.models import DataQualityCheck
from sam.storage.repositories import (
    DataQualityRepository,
    DocumentRepository,
    IngestionRunRepository,
    SaiRepository,
    SourceRepository,
)

log = get_logger("processing.quality")

# P3 gate: near-dup rate must stay under 2%.
DUP_RATE_WARN = 0.01
DUP_RATE_FAIL = 0.02
NEAR_DUP_JACCARD = 0.85
NEAR_DUP_WINDOW = 1000  # most recent documents scanned per check

# Daily cadence + slack. A stalled collector is a fail, not merely a warn.
FRESHNESS_WARN_HOURS = 26.0
FRESHNESS_FAIL_HOURS = 48.0

# Latest fetch vs trailing mean of prior successful runs.
VOLUME_WARN_RATIO = 0.5
VOLUME_MIN_HISTORY = 3  # runs needed before the check is meaningful

# Coverage is informational (most world news isn't about a 6-ticker universe),
# but zero links across a sizable corpus means the dictionary rotted.
COVERAGE_ROT_MIN_DOCS = 100

# Unlike resolution, enrichment applies to (almost) every document — a large
# backlog means the enrich stage stalled, which silently starves P5 signals.
ENRICHMENT_BACKLOG_WARN = 0.5  # warn when >50% of the corpus is unenriched

# A healthy daily chain leaves the panel at yesterday (lag 0). One missed run
# self-heals on the next (the watermark backfills); two+ days means broken.
SAI_STALE_WARN_DAYS = 2.0

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(slots=True)
class CheckOutcome:
    """One executed check, pre-persistence."""

    check_name: str
    status: str  # 'pass' | 'warn' | 'fail'
    source_name: str | None = None
    value: float | None = None
    threshold: float | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_row(self) -> DataQualityCheck:
        return DataQualityCheck(
            check_name=self.check_name,
            source_name=self.source_name,
            status=self.status,
            value=self.value,
            threshold=self.threshold,
            details=self.details,
        )


def _as_utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes for timezone-aware columns; pin to UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def _tokens(title: str) -> frozenset[str]:
    # Hyphens collapse ("start-up" == "startup"): the syndication variant
    # observed live differed only in hyphenation.
    return frozenset(_TOKEN_RE.findall(title.lower().replace("-", "")))


def near_duplicate_pairs(
    titles: list[tuple[int, str | None]], *, jaccard: float = NEAR_DUP_JACCARD
) -> list[tuple[int, int]]:
    """Document-id pairs whose titles are near-identical (token-set Jaccard).

    O(n^2) with a size prefilter — fine at the current window (<=1000 docs).
    MinHash replaces this when Reddit-scale volume lands.
    """
    tokenized = [(doc_id, _tokens(title)) for doc_id, title in titles if title]
    tokenized = [(doc_id, toks) for doc_id, toks in tokenized if len(toks) >= 3]
    pairs: list[tuple[int, int]] = []
    for i, (id_a, toks_a) in enumerate(tokenized):
        for id_b, toks_b in tokenized[i + 1 :]:
            # Jaccard >= t is impossible unless the smaller set is >= t of the larger.
            if min(len(toks_a), len(toks_b)) < jaccard * max(len(toks_a), len(toks_b)):
                continue
            union = len(toks_a | toks_b)
            if union and len(toks_a & toks_b) / union >= jaccard:
                pairs.append((min(id_a, id_b), max(id_a, id_b)))
    return pairs


def check_duplicate_rate(session: Session, *, window: int = NEAR_DUP_WINDOW) -> CheckOutcome:
    """Near-dup share of the recent window; the P3 '<2% dup rate' gate."""
    titles = DocumentRepository(session).recent_titles(limit=window)
    pairs = near_duplicate_pairs(titles)
    # Rate = extra members of dup groups / docs scanned (a pair costs one doc).
    extras = {b for _, b in pairs}
    rate = len(extras) / len(titles) if titles else 0.0
    status = "fail" if rate > DUP_RATE_FAIL else "warn" if rate > DUP_RATE_WARN else "pass"
    return CheckOutcome(
        check_name="duplicate_rate",
        status=status,
        value=round(rate, 6),
        threshold=DUP_RATE_FAIL,
        details={"window": len(titles), "pairs": [list(p) for p in pairs[:10]]},
    )


def check_freshness(session: Session, *, now: datetime | None = None) -> list[CheckOutcome]:
    """Hours since each source's last successful run (staleness alerting)."""
    now = now or datetime.now(tz=UTC)
    runs = IngestionRunRepository(session)
    outcomes: list[CheckOutcome] = []
    for source in SourceRepository(session).all():
        successes = runs.recent_successes_for(source.id, limit=1)
        if not successes or successes[0].finished_at is None:
            outcomes.append(
                CheckOutcome(
                    check_name="freshness",
                    source_name=source.name,
                    status="fail",
                    value=None,
                    threshold=FRESHNESS_FAIL_HOURS,
                    details={"reason": "no successful run recorded"},
                )
            )
            continue
        age_hours = (now - _as_utc(successes[0].finished_at)).total_seconds() / 3600
        status = (
            "fail"
            if age_hours > FRESHNESS_FAIL_HOURS
            else "warn"
            if age_hours > FRESHNESS_WARN_HOURS
            else "pass"
        )
        outcomes.append(
            CheckOutcome(
                check_name="freshness",
                source_name=source.name,
                status=status,
                value=round(age_hours, 2),
                threshold=FRESHNESS_FAIL_HOURS,
            )
        )
    return outcomes


def check_volume_anomaly(session: Session) -> list[CheckOutcome]:
    """Latest successful fetch size vs the source's trailing mean."""
    runs = IngestionRunRepository(session)
    outcomes: list[CheckOutcome] = []
    for source in SourceRepository(session).all():
        history = runs.recent_successes_for(source.id, limit=VOLUME_MIN_HISTORY + 7)
        if len(history) < VOLUME_MIN_HISTORY:
            outcomes.append(
                CheckOutcome(
                    check_name="volume_anomaly",
                    source_name=source.name,
                    status="pass",
                    value=None,
                    threshold=VOLUME_WARN_RATIO,
                    details={"reason": "insufficient history", "runs": len(history)},
                )
            )
            continue
        latest, *prior = history
        mean_prior = sum(r.rows_fetched for r in prior) / len(prior)
        # Zero trailing mean: nothing expected, nothing missing.
        ratio = 1.0 if mean_prior == 0 else latest.rows_fetched / mean_prior
        status = (
            "fail"
            if ratio == 0.0 and mean_prior > 0
            else "warn"
            if ratio < VOLUME_WARN_RATIO
            else "pass"
        )
        outcomes.append(
            CheckOutcome(
                check_name="volume_anomaly",
                source_name=source.name,
                status=status,
                value=round(ratio, 4),
                threshold=VOLUME_WARN_RATIO,
                details={"latest": latest.rows_fetched, "trailing_mean": round(mean_prior, 2)},
            )
        )
    return outcomes


def check_resolution_coverage(session: Session) -> CheckOutcome:
    """Share of resolved docs with >=1 entity link; backlog in details."""
    total, unresolved, with_links = DocumentRepository(session).resolution_stats()
    resolved = total - unresolved
    coverage = with_links / resolved if resolved else 0.0
    rotted = resolved >= COVERAGE_ROT_MIN_DOCS and with_links == 0
    return CheckOutcome(
        check_name="resolution_coverage",
        status="warn" if rotted else "pass",
        value=round(coverage, 4),
        threshold=None,
        details={"total": total, "unresolved": unresolved, "with_links": with_links},
    )


def check_enrichment_coverage(session: Session) -> CheckOutcome:
    """Share of docs enriched; warns on a stalled pipeline or zero output."""
    total, unenriched, with_sentiment = DocumentRepository(session).enrichment_stats()
    enriched = total - unenriched
    coverage = enriched / total if total else 0.0
    backlog_ratio = unenriched / total if total else 0.0
    rotted = enriched >= COVERAGE_ROT_MIN_DOCS and with_sentiment == 0
    stalled = total >= COVERAGE_ROT_MIN_DOCS and backlog_ratio > ENRICHMENT_BACKLOG_WARN
    return CheckOutcome(
        check_name="enrichment_coverage",
        status="warn" if (rotted or stalled) else "pass",
        value=round(coverage, 4),
        threshold=None,
        details={"total": total, "unenriched": unenriched, "with_sentiment": with_sentiment},
    )


def check_sai_freshness(session: Session, *, now: datetime | None = None) -> CheckOutcome:
    """Days the SAI panel trails yesterday (its expected last closed day).

    A missing panel is only a warning once a sizable linked corpus exists —
    before that, "no rows yet" is the honest state of a young project, not
    a failure.
    """
    now = now or datetime.now(tz=UTC)
    latest = SaiRepository(session).latest_date()
    _total, _unresolved, with_links = DocumentRepository(session).resolution_stats()
    if latest is None:
        sizable = with_links >= COVERAGE_ROT_MIN_DOCS
        return CheckOutcome(
            check_name="sai_freshness",
            status="warn" if sizable else "pass",
            value=None,
            threshold=SAI_STALE_WARN_DAYS,
            details={"reason": "no sai_daily rows", "linked_documents": with_links},
        )
    lag_days = float(((now.date() - timedelta(days=1)) - latest).days)
    return CheckOutcome(
        check_name="sai_freshness",
        status="warn" if lag_days >= SAI_STALE_WARN_DAYS else "pass",
        value=lag_days,
        threshold=SAI_STALE_WARN_DAYS,
        details={"latest_day": str(latest), "linked_documents": with_links},
    )


class DataQualityRunner:
    """Run all checks, persist their rows, and report the outcomes."""

    def __init__(self, session_factory: Callable[[], Session] | None = None) -> None:
        # Resolved lazily so tests can monkeypatch module-level default_session.
        self._session_factory = session_factory or default_session

    def run(self) -> list[CheckOutcome]:
        session = self._session_factory()
        try:
            outcomes: list[CheckOutcome] = [check_duplicate_rate(session)]
            outcomes.extend(check_freshness(session))
            outcomes.extend(check_volume_anomaly(session))
            outcomes.append(check_resolution_coverage(session))
            outcomes.append(check_enrichment_coverage(session))
            outcomes.append(check_sai_freshness(session))

            DataQualityRepository(session).record([o.to_row() for o in outcomes])
            session.commit()
        finally:
            session.close()

        failed = [o for o in outcomes if o.status == "fail"]
        warned = [o for o in outcomes if o.status == "warn"]
        log.info(
            "dq_run_complete",
            checks=len(outcomes),
            failed=[f"{o.check_name}:{o.source_name or '-'}" for o in failed],
            warned=[f"{o.check_name}:{o.source_name or '-'}" for o in warned],
        )
        return outcomes
