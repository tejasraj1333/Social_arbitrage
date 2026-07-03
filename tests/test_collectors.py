"""No-network unit tests for the recon collectors.

Live sources (RSS/Yahoo/HN) are proven by actually running them during recon;
here we mock their clients so CI stays offline and deterministic. Credentialed
sources (Reddit/Kaggle) are covered for both the live-shape and the
missing-credentials path via mocks.
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from sam.recon import collector_base as cb


@pytest.fixture(autouse=True)
def _isolate_data_dirs(tmp_path, monkeypatch):
    monkeypatch.setattr(cb, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(cb, "SAMPLE_DIR", tmp_path / "sample")


# --------------------------------------------------------------------------- RSS


def _fake_entry(i: int, link: str | None = None) -> dict[str, Any]:
    return {
        "title": f"Headline {i}",
        "link": link or f"https://news.example.com/{i}",
        "summary": f"<p>Body <b>{i}</b></p>",
        "published_parsed": time.gmtime(1700000000 + i * 60),
    }


def test_rss_fetch_maps_schema_and_dedupes(monkeypatch):
    from sam.collectors import rss_collector as rc

    feeds = [{"name": "A", "url": "a"}, {"name": "B", "url": "b"}]
    # Feed B repeats one of feed A's URLs -> should be deduped.
    feed_entries = {
        "a": [_fake_entry(0), _fake_entry(1)],
        "b": [_fake_entry(1), _fake_entry(2)],  # entry 1 duplicate by link
    }
    monkeypatch.setattr(rc, "_fetch_feed", lambda url: {"entries": feed_entries[url]})

    collector = rc.RSSCollector(feeds=feeds, target=100)
    records = collector.fetch()

    assert len(records) == 3  # 4 entries minus 1 duplicate
    first = records[0]
    assert set(first) == {"title", "summary", "url", "published_date", "source"}
    assert first["source"] == "A"
    assert "<" not in first["summary"]  # HTML stripped
    assert first["published_date"].endswith("+00:00")  # ISO-8601 UTC


def test_rss_respects_target(monkeypatch):
    from sam.collectors import rss_collector as rc

    entries = [_fake_entry(i) for i in range(50)]
    monkeypatch.setattr(rc, "_fetch_feed", lambda url: {"entries": entries})
    records = rc.RSSCollector(feeds=[{"name": "A", "url": "a"}], target=10).fetch()
    assert len(records) == 10


def test_rss_tolerates_empty_feed(monkeypatch):
    from sam.collectors import rss_collector as rc

    monkeypatch.setattr(rc, "_fetch_feed", lambda url: {"entries": [], "bozo_exception": "boom"})
    result = rc.RSSCollector(feeds=[{"name": "Dead", "url": "x"}]).run()
    assert result.status == "empty"
    assert result.record_count == 0


def test_rss_tolerates_network_failure_and_continues(monkeypatch):
    import httpx

    from sam.collectors import rss_collector as rc

    def fake_fetch(url):
        if url == "dead":
            raise httpx.ConnectTimeout("hang")  # timed-out feed must be skipped
        return {"entries": [_fake_entry(0), _fake_entry(1)]}

    monkeypatch.setattr(rc, "_fetch_feed", fake_fetch)
    feeds = [{"name": "Dead", "url": "dead"}, {"name": "Live", "url": "live"}]
    records = rc.RSSCollector(feeds=feeds).fetch()
    assert len(records) == 2  # the live feed still delivers


def test_rss_feed_download_enforces_timeout(monkeypatch):
    from sam.collectors import rss_collector as rc

    captured: dict[str, object] = {}

    class _FakeResponse:
        content = b"<rss></rss>"

        @staticmethod
        def raise_for_status() -> None:
            return None

    def fake_get(url, **kwargs):
        captured.update(kwargs, url=url)
        return _FakeResponse()

    monkeypatch.setattr(rc.httpx, "get", fake_get)
    rc._fetch_feed("https://example.com/feed")
    # feedparser must never fetch the URL itself (it has no timeout).
    assert captured["url"] == "https://example.com/feed"
    assert captured["timeout"] is rc._FETCH_TIMEOUT
    assert captured["follow_redirects"] is True


# ------------------------------------------------------------------------- Yahoo


def _fake_yf_frame():
    import pandas as pd

    dates = pd.DatetimeIndex(
        pd.to_datetime(["2026-01-02", "2026-01-03", "2026-01-06"]), name="Date"
    )
    tickers = ["AAPL", "MSFT"]
    fields = ["Adj Close", "Close", "High", "Low", "Open", "Volume"]
    cols = pd.MultiIndex.from_product([fields, tickers], names=["Price", "Ticker"])
    base = {"AAPL": 100.0, "MSFT": 200.0}
    rows = []
    for di in range(len(dates)):
        row = []
        for fld in fields:
            for tk in tickers:
                row.append(1_000_000 + di if fld == "Volume" else base[tk] + di)
        rows.append(row)
    frame = pd.DataFrame(rows, index=dates, columns=cols)
    # One missing bar -> the (MSFT, 2026-01-03) row must be dropped.
    frame.loc[dates[1], ("Close", "MSFT")] = float("nan")
    frame.loc[dates[1], ("Adj Close", "MSFT")] = float("nan")
    return frame, tickers


def test_yahoo_reshape_drops_nan_and_maps_schema(monkeypatch):
    from sam.collectors import yahoo_collector as yc

    frame, tickers = _fake_yf_frame()
    monkeypatch.setattr(yc.yf, "download", lambda *a, **k: frame)
    records = yc.YahooCollector(tickers=tickers).fetch()

    assert len(records) == 5  # 3 days x 2 tickers minus 1 NaN-close row
    rec = records[0]
    assert set(rec) == set(yc.YahooCollector.required_fields)
    assert isinstance(rec["volume"], int)
    assert all(r["close"] is not None for r in records)


def test_yahoo_saves_csv(monkeypatch):
    import pandas as pd

    from sam.collectors import yahoo_collector as yc

    frame, tickers = _fake_yf_frame()
    monkeypatch.setattr(yc.yf, "download", lambda *a, **k: frame)
    result = yc.YahooCollector(tickers=tickers).run()

    assert result.status == "ok"
    assert result.sample_path is not None and result.sample_path.endswith("yahoo_ohlcv.csv")
    saved = pd.read_csv(result.sample_path)
    assert list(saved.columns) == yc.YahooCollector.required_fields


# -------------------------------------------------------------------- Hacker News


def test_hn_fetch_hydrates_and_skips_dead_items(monkeypatch):
    from sam.collectors import hn_collector as hn

    ids = list(range(1, 6))
    items = {
        i: {
            "id": i,
            "title": f"Story {i}",
            "score": 10 + i,
            "descendants": i,
            "time": 1700000000 + i,
            "url": f"https://example.com/{i}",
            "by": "user",
            "type": "story",
        }
        for i in ids
    }
    items[3] = None  # deleted/dead item -> must be skipped

    def fake_get(self, client, url):
        if url.endswith("topstories.json"):
            return ids
        item_id = int(url.rsplit("/item/", 1)[1].split(".json")[0])
        return items[item_id]

    monkeypatch.setattr(hn.HackerNewsCollector, "_get", fake_get)
    records = hn.HackerNewsCollector(limit=5).fetch()

    assert len(records) == 4  # 5 ids minus 1 dead item
    rec = records[0]
    assert {"title", "score", "comments", "timestamp", "url"}.issubset(rec)
    assert rec["comments"] == 1  # mapped from descendants


def test_hn_respects_limit(monkeypatch):
    from sam.collectors import hn_collector as hn

    ids = list(range(1, 21))

    def fake_get(self, client, url):
        if url.endswith("topstories.json"):
            return ids
        item_id = int(url.rsplit("/item/", 1)[1].split(".json")[0])
        return {"id": item_id, "title": "t", "score": 1, "descendants": 0, "time": 1, "url": "u"}

    monkeypatch.setattr(hn.HackerNewsCollector, "_get", fake_get)
    assert len(hn.HackerNewsCollector(limit=7).fetch()) == 7


# ------------------------------------------------------------------------ Reddit


class _FakeAuthor:
    def __init__(self, name):
        self.name = name


class _FakeSubmission:
    def __init__(self, i, author="u"):
        self.id = f"id{i}"
        self.title = f"Post {i}"
        self.selftext = "" if i % 2 else f"body {i}"  # link posts have empty body
        self.score = 100 + i
        self.num_comments = i
        self.created_utc = 1700000000.0 + i
        self.author = _FakeAuthor(author) if author else None


class _FakeSubreddit:
    def __init__(self, n=3):
        self._n = n

    def hot(self, limit):
        return (_FakeSubmission(i) for i in range(min(limit, self._n)))


class _FakeReddit:
    def subreddit(self, name):
        return _FakeSubreddit()


def _creds(client_id="cid", client_secret="secret"):
    from types import SimpleNamespace

    return SimpleNamespace(
        reddit=SimpleNamespace(client_id=client_id, client_secret=client_secret, user_agent="ua")
    )


def test_reddit_fetch_maps_schema(monkeypatch):
    from sam.collectors import reddit_collector as rc

    monkeypatch.setattr(rc, "get_settings", _creds)
    monkeypatch.setattr(rc.RedditCollector, "_client", lambda self, creds: _FakeReddit())

    records = rc.RedditCollector(subreddits=["stocks", "investing"], post_limit=6).fetch()
    assert len(records) == 6  # 3 per subreddit x 2 subreddits
    rec = records[0]
    assert {
        "title",
        "body",
        "score",
        "comments",
        "created_utc",
        "subreddit",
        "author",
    }.issubset(rec)
    assert rec["subreddit"] == "stocks"


def test_reddit_needs_credentials(monkeypatch):
    from sam.collectors import reddit_collector as rc

    monkeypatch.setattr(rc, "get_settings", lambda: _creds(client_id="", client_secret=""))
    result = rc.RedditCollector(subreddits=["stocks"]).run()
    assert result.status == "needs_credentials"
    assert result.record_count == 0


# ------------------------------------------------------------------------ Kaggle


class _FakeDataset:
    def __init__(self, ref):
        self.ref = ref
        self.title = f"Title {ref}"
        self.size = "12MB"
        self.licenseName = "CC0-1.0"
        self.lastUpdated = "2026-01-01"
        self.downloadCount = 500
        self.voteCount = 42
        self.usabilityRating = 0.88


class _FakeKaggleApi:
    def __init__(self, by_term):
        self._by_term = by_term

    def dataset_list(self, search, sort_by=None):
        return self._by_term.get(search, [])


def test_kaggle_fetch_maps_metadata_and_dedupes(monkeypatch):
    from sam.collectors import kaggle_evaluator as ke

    by_term = {
        "reddit": [_FakeDataset("a/r1"), _FakeDataset("a/r2")],
        "finance": [_FakeDataset("a/r2"), _FakeDataset("a/f1")],  # r2 dup across terms
    }
    monkeypatch.setattr(ke.KaggleEvaluator, "_api", lambda self: _FakeKaggleApi(by_term))
    records = ke.KaggleEvaluator(search_terms=["reddit", "finance"], per_term=20).fetch()

    assert len(records) == 3  # 4 minus 1 cross-term duplicate
    rec = records[0]
    assert set(ke.KaggleEvaluator.required_fields).issubset(rec)
    assert rec["license"] == "CC0-1.0"
    assert rec["search_term"] == "reddit"


def test_kaggle_respects_per_term(monkeypatch):
    from sam.collectors import kaggle_evaluator as ke

    by_term = {"reddit": [_FakeDataset(f"a/{i}") for i in range(10)]}
    monkeypatch.setattr(ke.KaggleEvaluator, "_api", lambda self: _FakeKaggleApi(by_term))
    records = ke.KaggleEvaluator(search_terms=["reddit"], per_term=4).fetch()
    assert len(records) == 4


def test_kaggle_needs_credentials(monkeypatch):
    from sam.collectors import kaggle_evaluator as ke
    from sam.core.errors import CredentialsMissing

    def boom(self):
        raise CredentialsMissing("no token")

    monkeypatch.setattr(ke.KaggleEvaluator, "_api", boom)
    result = ke.KaggleEvaluator(search_terms=["reddit"]).run()
    assert result.status == "needs_credentials"
    assert result.record_count == 0


def test_kaggle_bridges_settings_to_env(monkeypatch):
    """SAM_KAGGLE__* settings must reach the env vars the kaggle client reads."""
    import os
    from types import SimpleNamespace

    from sam.collectors import kaggle_evaluator as ke

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    settings = SimpleNamespace(kaggle=SimpleNamespace(username="sam-user", key="sam-key"))
    monkeypatch.setattr(ke, "get_settings", lambda: settings)

    ke.KaggleEvaluator._export_credentials()
    assert os.environ["KAGGLE_USERNAME"] == "sam-user"
    assert os.environ["KAGGLE_KEY"] == "sam-key"

    # Explicit env vars already set by the user must win (setdefault semantics).
    monkeypatch.setenv("KAGGLE_USERNAME", "explicit")
    ke.KaggleEvaluator._export_credentials()
    assert os.environ["KAGGLE_USERNAME"] == "explicit"
