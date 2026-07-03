"""Production Yahoo Finance ingestion collector (market labels).

Fetch is delegated to the recon YahooCollector (MultiIndex reshape, NaN-bar
dropping). Persists into market_data keyed (entity_id, date) with DO UPDATE —
vendor restatements (splits/dividends adjusting adj_close) must win.

Incremental by default (small trailing window; the upsert makes overlap free);
`backfill=True` pulls the full configured period (e.g. 1y) instead.

Only tickers present in the entities table are persisted — the universe is
curated via `sam seed` (config/sources.yaml), unknown tickers are logged and
skipped, never auto-created.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date
from typing import Any

from sqlalchemy.orm import Session

from sam.collectors.yahoo_collector import YahooCollector as _YahooFetcher
from sam.core.logging import get_logger
from sam.ingestion.base import Collector
from sam.recon.sources import load_sources
from sam.storage.repositories import EntityRepository, MarketDataRepository

_INCREMENTAL_PERIOD = "7d"  # trailing window; overlap is free thanks to the upsert


class YahooIngestionCollector(Collector):
    source_type = "yahoo"
    source_name = "yahoo"
    config_ref = "config/sources.yaml:yahoo"

    def __init__(
        self,
        session: Session,
        source_id: int,
        tickers: list[str] | None = None,
        backfill: bool = False,
    ) -> None:
        self.session = session
        self.source_id = source_id
        self.log = get_logger(f"ingestion.{self.source_name}")
        cfg = load_sources().get("yahoo", {})
        period = cfg.get("period", "1y") if backfill else _INCREMENTAL_PERIOD
        self._fetcher = _YahooFetcher(tickers=tickers, period=period)

    def fetch(self) -> Iterable[dict[str, Any]]:
        return self._fetcher.fetch()

    def normalize(self, raw: dict[str, Any]) -> dict[str, Any]:
        return {
            "ticker": raw.get("ticker"),
            "date": date.fromisoformat(str(raw.get("date"))),
            "open": raw.get("open"),
            "high": raw.get("high"),
            "low": raw.get("low"),
            "close": raw.get("close"),
            "adj_close": raw.get("adj_close"),
            "volume": raw.get("volume"),
        }

    def persist(self, documents: Iterable[dict[str, Any]]) -> int:
        bars = list(documents)
        ids = EntityRepository(self.session).by_ticker()

        known: list[dict[str, Any]] = []
        skipped: set[str] = set()
        for bar in bars:
            entity_id = ids.get(bar["ticker"])
            if entity_id is None:
                skipped.add(str(bar["ticker"]))
                continue
            row = dict(bar, entity_id=entity_id)
            row.pop("ticker")
            known.append(row)

        if skipped:
            self.log.warning(
                "unknown_tickers_skipped",
                tickers=sorted(skipped),
                hint="add them to config/sources.yaml universe and run `sam seed`",
            )
        written = MarketDataRepository(self.session).upsert_many(known)
        self.log.info("bars_persisted", source=self.source_name, written=written)
        return written
