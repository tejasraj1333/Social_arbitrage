"""Production-collector tests: mocked fetch, real (SQLite) persistence.

The key property under test is end-to-end idempotency: running the same
collector twice against the same upstream data inserts rows exactly once.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from sqlalchemy import select
from sqlalchemy.orm import Session

from sam.ingestion.base import parse_utc
from sam.ingestion.hackernews import HackerNewsIngestionCollector
from sam.ingestion.rss import RSSIngestionCollector
from sam.ingestion.yahoo import YahooIngestionCollector
from sam.storage.models import Document, MarketData
from sam.storage.repositories import EntityRepository, SourceRepository

UNIVERSE = [{"ticker": "AAPL", "name": "Apple Inc."}, {"ticker": "MSFT", "name": "Microsoft"}]


def _source_id(session: Session, type_: str, name: str) -> int:
    return SourceRepository(session).get_or_create(type_, name).id


# ----------------------------------------------------------------- parse_utc


def test_parse_utc_pins_naive_to_utc() -> None:
    assert parse_utc("2026-07-01T10:00:00") == datetime(2026, 7, 1, 10, tzinfo=UTC)
    assert parse_utc("2026-07-01T10:00:00Z") == datetime(2026, 7, 1, 10, tzinfo=UTC)
    assert parse_utc("2026-07-01T12:00:00+02:00") == datetime(2026, 7, 1, 10, tzinfo=UTC)
    assert parse_utc(None) is None
    assert parse_utc("garbage") is None


# ----------------------------------------------------------------------- RSS

RSS_RAW = [
    {
        "title": "Tech rout intensifies",
        "summary": "Global stocks sold off.",
        "url": "https://news.example.com/1",
        "published_date": "2026-07-02T08:00:00+00:00",
        "source": "CNBC",
    },
    {
        "title": "Fed holds rates",
        "summary": "No change.",
        "url": "https://news.example.com/2",
        "published_date": None,  # missing publish date must not crash
        "source": "MarketWatch",
    },
]


def test_rss_ingest_normalizes_and_is_idempotent(db_session: Session, monkeypatch) -> None:
    sid = _source_id(db_session, "rss", "rss")
    collector = RSSIngestionCollector(db_session, sid)
    monkeypatch.setattr(collector, "fetch", lambda: list(RSS_RAW))

    assert collector.run() == 2
    assert collector.run() == 0  # same upstream data -> no-op

    docs = db_session.execute(select(Document).order_by(Document.external_id)).scalars().all()
    assert len(docs) == 2
    first = docs[0]
    assert first.source_id == sid
    assert first.url == "https://news.example.com/1"
    assert first.raw_text == "Global stocks sold off."
    assert first.published_at is not None
    assert first.engagement == {"feed": "CNBC"}
    assert len(first.content_hash) == 64
    assert docs[1].published_at is None


def test_rss_edited_title_is_new_document(db_session: Session, monkeypatch) -> None:
    sid = _source_id(db_session, "rss", "rss")
    collector = RSSIngestionCollector(db_session, sid)
    monkeypatch.setattr(collector, "fetch", lambda: list(RSS_RAW))
    collector.run()

    edited = [dict(RSS_RAW[0], title="Tech rout intensifies (updated)"), RSS_RAW[1]]
    monkeypatch.setattr(collector, "fetch", lambda: edited)
    assert collector.run() == 1  # only the edited item re-inserts


# ------------------------------------------------------------------------ HN

HN_RAW = [
    {
        "id": 101,
        "title": "Show HN: something",
        "score": 42,
        "comments": 7,
        "timestamp": 1_780_000_000,
        "url": "https://example.com/x",
        "by": "alice",
        "type": "story",
    },
    {
        "id": 102,
        "title": "Ask HN: no url",
        "score": 5,
        "comments": 1,
        "timestamp": 1_780_000_100,
        "url": None,
        "by": "bob",
        "type": "story",
    },
]


def test_hn_ingest_engagement_snapshot_and_idempotency(db_session: Session, monkeypatch) -> None:
    sid = _source_id(db_session, "hackernews", "hackernews")
    collector = HackerNewsIngestionCollector(db_session, sid)
    monkeypatch.setattr(collector, "fetch", lambda: list(HN_RAW))

    assert collector.run() == 2

    # Re-fetch with changed engagement: same identity -> no new rows,
    # first-seen snapshot preserved (point-in-time).
    hotter = [dict(item, score=item["score"] * 10) for item in HN_RAW]
    monkeypatch.setattr(collector, "fetch", lambda: hotter)
    assert collector.run() == 0

    doc = db_session.execute(select(Document).where(Document.external_id == "101")).scalar_one()
    assert doc.engagement == {"score": 42, "comments": 7}
    assert doc.author == "alice"
    # SQLite returns DateTime(timezone=True) naive; the UTC instant must match.
    stored = doc.published_at
    assert stored is not None
    if stored.tzinfo is None:
        stored = stored.replace(tzinfo=UTC)
    assert stored == datetime.fromtimestamp(1_780_000_000, tz=UTC)


# --------------------------------------------------------------------- Yahoo

YAHOO_RAW = [
    {
        "ticker": "AAPL",
        "date": "2026-07-01",
        "open": 10.0,
        "high": 11.0,
        "low": 9.0,
        "close": 10.5,
        "adj_close": 10.5,
        "volume": 100,
    },
    {
        "ticker": "MSFT",
        "date": "2026-07-01",
        "open": 20.0,
        "high": 21.0,
        "low": 19.0,
        "close": 20.5,
        "adj_close": 20.5,
        "volume": 200,
    },
    {
        "ticker": "ZZZZ",  # not in the curated universe -> skipped, not created
        "date": "2026-07-01",
        "open": 1.0,
        "high": 1.0,
        "low": 1.0,
        "close": 1.0,
        "adj_close": 1.0,
        "volume": 1,
    },
]


@pytest.fixture
def yahoo(db_session: Session, monkeypatch) -> YahooIngestionCollector:
    EntityRepository(db_session).seed(UNIVERSE)
    sid = _source_id(db_session, "yahoo", "yahoo")
    collector = YahooIngestionCollector(db_session, sid)
    monkeypatch.setattr(collector, "fetch", lambda: [dict(r) for r in YAHOO_RAW])
    return collector


def test_yahoo_ingest_skips_unknown_tickers(yahoo: YahooIngestionCollector, db_session) -> None:
    assert yahoo.run() == 2  # ZZZZ skipped
    bars = db_session.execute(select(MarketData)).scalars().all()
    assert len(bars) == 2


def test_yahoo_reingest_applies_restatements(
    yahoo: YahooIngestionCollector, db_session, monkeypatch
) -> None:
    yahoo.run()
    restated = [dict(YAHOO_RAW[0], adj_close=9.9), YAHOO_RAW[1]]
    monkeypatch.setattr(yahoo, "fetch", lambda: restated)
    yahoo.run()

    ids = EntityRepository(db_session).by_ticker()
    bar = db_session.execute(
        select(MarketData).where(MarketData.entity_id == ids["AAPL"])
    ).scalar_one()
    assert bar.adj_close == 9.9  # restatement won
    assert len(db_session.execute(select(MarketData)).scalars().all()) == 2  # no dupes


def test_yahoo_backfill_uses_configured_period(db_session: Session) -> None:
    sid = _source_id(db_session, "yahoo", "yahoo")
    incremental = YahooIngestionCollector(db_session, sid)
    backfill = YahooIngestionCollector(db_session, sid, backfill=True)
    assert incremental._fetcher.period == "7d"
    assert backfill._fetcher.period == "1y"  # from config/sources.yaml
