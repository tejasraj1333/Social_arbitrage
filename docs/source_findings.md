# Source Findings Report — Phase 1 (Source Recon)

Conclusions from running each candidate source. For every source we answer the five
recon questions: (1) historical data obtainable? (2) live data obtainable?
(3) expected daily volume? (4) main risks? (5) recommended for production?

Measured figures come from the live recon run captured in
[`source_scorecard.md`](source_scorecard.md) / `data/sample/source_scorecard.csv`.
Legal basis for each source is in [`legal_register.md`](legal_register.md).

## Verdict summary

| Source | Historical | Live | ~Daily volume (current scope) | Production verdict |
| --- | --- | --- | --- | --- |
| News RSS | Shallow (current feed only) | ✅ Yes | ~2–60 items/day (scales with feed count) | ✅ Recommended (headline/link only) |
| Yahoo Finance | ✅ Deep (years daily) | ✅ Yes (daily) | 5 bars/trading day (5-ticker universe) | ⚠️ Research only — migrate to licensed feed |
| Hacker News | ✅ Yes (via item IDs) | ✅ Yes | ~40/day (top-100 churn) | ✅ Recommended |
| Reddit | ⚠️ Limited (≈1000/listing) | ✅ Yes (pending creds) | Hundreds–thousands/day (3 subs) | ✅ Recommended (ML use needs agreement) |
| Kaggle | ✅ Yes (static snapshots) | ❌ No (reference data) | N/A — batch backfill | ✅ Recommended for backfill (per-license) |

---

## News RSS

- **Measured:** status `ok`, 100 records, schema completeness 1.0, freshness ~0.1h.
- **Historical?** Shallow. A feed exposes only its current window (latest ~10–30
  items, hours to a few days). No deep archive via RSS.
- **Live?** Yes — near-real-time; new headlines appear within minutes.
- **Expected daily volume:** ~2–60 curated top-items/day at the current 4-feed config;
  scales roughly linearly as more publisher feeds are added.
- **Main risks:** Feed URLs change or die (Reuters already retired its public RSS);
  ToS limits us to headline + summary + link (no full text); duplicate articles across
  feeds (handled via URL dedup).
- **Recommended for production?** ✅ Yes — cheap, keyless, fresh attention/news signal.
  Persist headline/summary/link only; add conditional GET + caching.

## Yahoo Finance

- **Measured:** status `ok`, 1255 rows (5 tickers × ~251 trading days), completeness 1.0.
- **Historical?** Yes — deep daily history (years) including Adjusted Close.
- **Live?** Yes for daily bars (and delayed intraday); not a true real-time feed.
- **Expected daily volume:** 5 OHLCV bars per trading day for the 5-ticker universe
  (~105/month); scales linearly with universe size.
- **Main risks:** Unofficial API with undocumented rate limits and no SLA — can break
  or throttle without notice; ToS restricts to personal/non-commercial use.
- **Recommended for production?** ⚠️ Research/backtesting only. Migrate price data to a
  licensed market-data vendor before production/commercial use.

## Hacker News

- **Measured:** status `ok`, 100 records, schema completeness ≈0.99 (Ask/Show-HN
  posts legitimately lack a URL), freshness ~1.5h.
- **Historical?** Yes — every item has a monotonic integer ID, so history can be walked
  backward by ID (or via the public BigQuery HN dataset). The `topstories` endpoint
  itself is current-only.
- **Live?** Yes — official keyless Firebase API, updated continuously.
- **Expected daily volume:** ~40/day from top-100 turnover (~1100–1250/month estimated).
  The full firehose (all new stories/comments) is far larger if needed.
- **Main risks:** One HTTP request per item (latency at scale — batch/concurrency
  needed); content is user-generated (quality/noise); Ask/Show-HN posts have no URL.
- **Recommended for production?** ✅ Yes — official, free, reliable early-attention
  signal for tech/consumer products.

## Reddit

- **Measured:** status `needs_credentials` (collector built + unit-tested; not yet run
  live — no API credentials in this environment).
- **Historical?** Limited via the API — listings cap at ~1000 items and skew recent.
  Deep history needs archives (Pushshift-style, now restricted) or a commercial deal.
- **Live?** Yes, once credentials are supplied — `hot`/`new` listings are near-real-time.
- **Expected daily volume:** Hundreds to low-thousands of posts/day across
  `wallstreetbets` + `stocks` + `investing` (comments add an order of magnitude more).
- **Main risks:** OAuth app + 100 QPM limits; 2023 ToS restricts commercial/ML use and
  requires honoring deletions; author handles are PII-adjacent.
- **Recommended for production?** ✅ Yes — a primary retail-sentiment signal — **but**
  secure a Reddit data agreement before training models on the content, and implement
  deletion handling. **Next action: add credentials and run live to confirm.**

## Kaggle Datasets

- **Measured:** status `needs_credentials` (evaluator built + unit-tested; metadata-only,
  no downloads — no API token in this environment).
- **Historical?** Yes — datasets are static historical snapshots, ideal for backfilling
  training data (e.g. historical Reddit/finance/news corpora).
- **Live?** No — reference/batch source, not a streaming signal.
- **Expected daily volume:** N/A (one-off/periodic batch ingestion, not a daily stream).
- **Main risks:** **Per-dataset licensing** is the dominant constraint — redistribution
  and commercial use vary by dataset; some are non-commercial or custom-licensed.
  Dataset freshness/maintenance also varies widely.
- **Recommended for production?** ✅ Yes, as a **backfill/bootstrap** source — gated on a
  per-dataset license allowlist (CC0/CC-BY first). Not a live signal.
  **Next action: add API token and run the metadata evaluator live.**

---

## Phase-1 outcome

Three keyless sources (News RSS, Yahoo Finance, Hacker News) were **proven live** —
each fetched a clean sample at schema completeness ≥ 0.99, well above the "one week of
usable data" bar. Two credentialed sources (Reddit, Kaggle) are **built and
unit-tested**, degrading cleanly to `needs_credentials`; they are one `.env` edit away
from a live run. Foundation for Phase 2 (ingestion + storage) is validated.
