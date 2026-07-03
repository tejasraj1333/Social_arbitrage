"""Data-quality framework tests (Phase 3 / M5).

Pure near-dup math is tested directly; the checks run on in-memory SQLite
with fabricated ingestion history; the runner test proves rows are persisted.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from sam.processing.quality import (
    DataQualityRunner,
    check_duplicate_rate,
    check_freshness,
    check_resolution_coverage,
    check_volume_anomaly,
    near_duplicate_pairs,
)
from sam.storage.models import DataQualityCheck
from sam.storage.repositories import (
    DocumentEntityRepository,
    DocumentRepository,
    EntityRepository,
    IngestionRunRepository,
    SourceRepository,
)

# Real syndication case observed live: same story, hyphenation differs.
NEAR_DUP_A = "Nvidia offers start-up customers chance to swap compute power for revenue share"
NEAR_DUP_B = "Nvidia offers startup customers chance to swap compute power for revenue share"


def _doc(hash_: str, title: str) -> dict[str, object]:
    return {
        "source_id": 1,
        "external_id": hash_[:8],
        "url": f"https://example.com/{hash_[:8]}",
        "author": None,
        "title": title,
        "raw_text": None,
        "lang": None,
        "content_hash": hash_,
        "published_at": None,
        "engagement": {},
    }


# ---------------------------------------------------------------- pure math


def test_near_duplicate_pairs_finds_syndicated_headline() -> None:
    titles: list[tuple[int, str | None]] = [
        (1, NEAR_DUP_A),
        (2, NEAR_DUP_B),
        (3, "Fed leaves rates unchanged"),
        (4, None),
    ]
    assert near_duplicate_pairs(titles) == [(1, 2)]


def test_near_duplicate_pairs_ignores_short_and_dissimilar_titles() -> None:
    titles: list[tuple[int, str | None]] = [
        (1, "Markets today"),  # <3 tokens: skipped
        (2, "Markets today again"),
        (3, "Oil prices fall on supply data"),
        (4, "Gold prices rise on safe-haven demand"),
    ]
    assert near_duplicate_pairs(titles) == []


# ------------------------------------------------------------------- checks


def test_duplicate_rate_pass_and_fail(db_session: Session) -> None:
    SourceRepository(db_session).get_or_create("rss", "rss")
    docs = DocumentRepository(db_session)
    docs.upsert_many(
        [
            _doc("a" * 64, NEAR_DUP_A),
            _doc("b" * 64, NEAR_DUP_B),
            _doc("c" * 64, "Fed leaves rates unchanged this quarter"),
        ]
    )
    db_session.commit()

    outcome = check_duplicate_rate(db_session)
    assert outcome.check_name == "duplicate_rate"
    assert outcome.value is not None and abs(outcome.value - 1 / 3) < 1e-6
    assert outcome.status == "fail"  # 33% >> 2% gate
    assert outcome.details["pairs"] == [[1, 2]]

    # Larger clean window dilutes the rate below the gate.
    docs.upsert_many(
        [_doc(f"{i:064x}", f"Unique headline number {i} about topic {i}") for i in range(120)]
    )
    db_session.commit()
    assert check_duplicate_rate(db_session).status == "pass"


def test_freshness_statuses(db_session: Session) -> None:
    src_fresh = SourceRepository(db_session).get_or_create("rss", "rss")
    src_stale = SourceRepository(db_session).get_or_create("yahoo", "yahoo")
    src_dead = SourceRepository(db_session).get_or_create("hackernews", "hackernews")
    runs = IngestionRunRepository(db_session)

    now = datetime.now(tz=UTC)
    fresh = runs.start(src_fresh.id)
    runs.finish(fresh, status="success", rows_fetched=10)
    fresh.finished_at = now - timedelta(hours=1)

    stale = runs.start(src_stale.id)
    runs.finish(stale, status="success", rows_fetched=10)
    stale.finished_at = now - timedelta(hours=30)

    dead = runs.start(src_dead.id)
    runs.finish(dead, status="error", error="boom")  # no success ever
    db_session.commit()

    by_source = {o.source_name: o for o in check_freshness(db_session, now=now)}
    assert by_source["rss"].status == "pass"
    assert by_source["yahoo"].status == "warn"  # 30h > 26h warn threshold
    assert by_source["hackernews"].status == "fail"
    assert by_source["hackernews"].details["reason"] == "no successful run recorded"


def test_volume_anomaly_statuses(db_session: Session) -> None:
    source = SourceRepository(db_session).get_or_create("rss", "rss")
    runs = IngestionRunRepository(db_session)

    # Trailing history of ~100 rows/run, then a collapse to 10 (warn).
    for fetched in (100, 110, 90, 10):
        run = runs.start(source.id)
        runs.finish(run, status="success", rows_fetched=fetched)
    db_session.commit()

    (outcome,) = check_volume_anomaly(db_session)
    assert outcome.status == "warn"
    assert outcome.value is not None and outcome.value < 0.5

    # A zero fetch against a non-zero trailing mean is a hard fail.
    run = runs.start(source.id)
    runs.finish(run, status="success", rows_fetched=0)
    db_session.commit()
    (outcome,) = check_volume_anomaly(db_session)
    assert outcome.status == "fail"


def test_volume_anomaly_needs_history(db_session: Session) -> None:
    source = SourceRepository(db_session).get_or_create("rss", "rss")
    runs = IngestionRunRepository(db_session)
    runs.finish(runs.start(source.id), status="success", rows_fetched=50)
    db_session.commit()

    (outcome,) = check_volume_anomaly(db_session)
    assert outcome.status == "pass"
    assert outcome.details["reason"] == "insufficient history"


def test_resolution_coverage_counts(db_session: Session) -> None:
    SourceRepository(db_session).get_or_create("rss", "rss")
    EntityRepository(db_session).seed([{"ticker": "NVDA", "name": "NVIDIA Corporation"}])
    docs = DocumentRepository(db_session)
    docs.upsert_many([_doc("a" * 64, "Nvidia rises"), _doc("b" * 64, "Fed holds")])
    now = datetime.now(tz=UTC)
    docs.mark_resolved([1, 2], at=now)
    DocumentEntityRepository(db_session).upsert_many(
        [
            {
                "document_id": 1,
                "entity_id": 1,
                "confidence": 0.8,
                "method": "alias",
                "resolved_at": now,
            }
        ]
    )
    db_session.commit()

    outcome = check_resolution_coverage(db_session)
    assert outcome.status == "pass"
    assert outcome.value == 0.5  # 1 of 2 resolved docs holds a link
    assert outcome.details == {"total": 2, "unresolved": 0, "with_links": 1}


# ------------------------------------------------------------------- runner


def test_dq_runner_persists_check_rows(db_session: Session) -> None:
    source = SourceRepository(db_session).get_or_create("rss", "rss")
    runs = IngestionRunRepository(db_session)
    runs.finish(runs.start(source.id), status="success", rows_fetched=5)
    DocumentRepository(db_session).upsert_many([_doc("a" * 64, "Nvidia rises on AI optimism")])
    db_session.commit()

    outcomes = DataQualityRunner(session_factory=lambda: db_session).run()
    names = {o.check_name for o in outcomes}
    assert names == {"duplicate_rate", "freshness", "volume_anomaly", "resolution_coverage"}

    rows = db_session.execute(select(DataQualityCheck)).scalars().all()
    assert len(rows) == len(outcomes)
    assert {row.status for row in rows} <= {"pass", "warn", "fail"}
