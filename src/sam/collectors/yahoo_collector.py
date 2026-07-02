"""Yahoo Finance recon collector (keyless).

Pulls daily OHLCV + Adjusted Close for the configured universe via yfinance and
reshapes the wide (Price x Ticker) frame into long per-(ticker, day) rows. Unlike
the text sources this saves a CSV (the natural shape for price bars); the saved
"sample" is the full ~1y history, which is the proof artifact.

Record schema: ticker, date (ISO), open, high, low, close, adj_close, volume.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd
import yfinance as yf

from sam.recon import collector_base as cb
from sam.recon.collector_base import ReconCollector
from sam.recon.sources import load_sources

_RENAME = {
    "Date": "date",
    "Ticker": "ticker",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Adj Close": "adj_close",
    "Volume": "volume",
}


def _f(value: Any) -> float | None:
    """Coerce to float, mapping NaN/None to None."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(out) else out


def _i(value: Any) -> int | None:
    f = _f(value)
    return None if f is None else int(f)


class YahooCollector(ReconCollector):
    source_name = "yahoo"
    required_fields = ["ticker", "date", "open", "high", "low", "close", "adj_close", "volume"]
    timestamp_field = "date"

    def __init__(
        self,
        tickers: list[str] | None = None,
        period: str | None = None,
        interval: str | None = None,
    ) -> None:
        super().__init__()
        cfg = load_sources().get("yahoo", {})
        self.tickers = tickers if tickers is not None else cfg.get("universe", [])
        self.period = period or cfg.get("period", "1y")
        self.interval = interval or cfg.get("interval", "1d")

    def fetch(self) -> list[dict[str, Any]]:
        frame = yf.download(
            self.tickers,
            period=self.period,
            interval=self.interval,
            auto_adjust=False,  # keep both Close and Adj Close
            group_by="column",
            progress=False,
            threads=True,
        )
        if frame is None or frame.empty:
            self.log.warning("yahoo_empty_frame", tickers=self.tickers)
            return []
        return self._reshape(frame)

    def _reshape(self, frame: pd.DataFrame) -> list[dict[str, Any]]:
        if isinstance(frame.columns, pd.MultiIndex):
            long = frame.stack(level="Ticker", future_stack=True)
            long.index = long.index.set_names(["Date", "Ticker"])
            long = long.reset_index()
        else:  # single-ticker fallback (yfinance can return a flat frame)
            long = frame.reset_index()
            long["Ticker"] = self.tickers[0] if self.tickers else "UNKNOWN"

        long = long.rename(columns=_RENAME)
        records: list[dict[str, Any]] = []
        for row in long.to_dict("records"):
            close = _f(row.get("close"))
            if close is None:  # non-trading day / missing bar
                continue
            raw_date = row.get("date")
            date_str = raw_date.date().isoformat() if hasattr(raw_date, "date") else str(raw_date)
            records.append(
                {
                    "ticker": row.get("ticker"),
                    "date": date_str,
                    "open": _f(row.get("open")),
                    "high": _f(row.get("high")),
                    "low": _f(row.get("low")),
                    "close": close,
                    "adj_close": _f(row.get("adj_close")),
                    "volume": _i(row.get("volume")),
                }
            )
        records.sort(key=lambda r: (r["ticker"], r["date"]))
        return records

    def save_sample(self, records: Any) -> tuple[str | None, str | None]:
        """Override: persist price bars as CSV (raw == sample == full history)."""
        if not records:
            return None, None
        cb.RAW_DIR.mkdir(parents=True, exist_ok=True)
        cb.SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = cb.RAW_DIR / "yahoo_ohlcv.csv"
        sample_path = cb.SAMPLE_DIR / "yahoo_ohlcv.csv"
        df = pd.DataFrame.from_records(list(records), columns=self.required_fields)
        df.to_csv(raw_path, index=False)
        df.to_csv(sample_path, index=False)
        return str(raw_path), str(sample_path)
