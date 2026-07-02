# Social Arbitrage Model — Phase 1 Execution Blueprint (Data-First)

> Built around the **70/30 data-first philosophy**: ~70% of effort on data
> acquisition/discovery/quality/engineering, ~30% on modeling/forecasting/LLM/dashboards.
>
> **Framing principle:** the moat is a clean, point-in-time, entity-resolved social
> dataset that nobody else has bothered to assemble correctly. Models are commodities;
> a leak-free labeled panel of "attention → outcome" is not. Every decision optimizes for that.

---

## STEP 1 — Implementation Roadmap

Realistic for a **solo developer, ~15–20 hrs/week, portfolio-grade**. Total ≈ 20 weeks;
the 12-week plan in Step 7 is the intensive core (MVP through first validated signal).

| Phase | Weeks | Focus (data:model) | Deliverable | Gate to pass |
|---|---|---|---|---|
| **P0 Foundations** | 0 (done) | — | M0 scaffold | `docker compose up` green |
| **P1 Source Recon** | 1–2 | 100% data | Source scorecard, legal log, 3 spike collectors | Each source proven to yield ≥1 week of data |
| **P2 Ingestion Backbone** | 3–5 | 95% data | Reddit+RSS+Yahoo+Kaggle → raw lake, idempotent | 30 days backfilled, dedup working |
| **P3 Entity Resolution & Quality** | 5–7 | 90% data | Ticker mapping, DQ framework, bot/spam filters | ≥90% entity precision; <2% dup rate |
| **P4 Enrichment & Feature Store** | 7–9 | 80% data | Sentiment+embeddings+topics, point-in-time features | Features rebuild deterministically |
| **P5 SAI + Signal Validation** | 9–11 | 60% data | SAI index + backtest harness | **KILL-GATE: significant forward IC** |
| **P6 Modeling** | 11–13 | 30% data | XGBoost/LightGBM forecasts | Beats naive baseline OOS |
| **P7 Reasoning & Reporting** | 13–15 | LLM | Grounded narrative reports | Zero hallucinated numbers |
| **P8 Serving & Portfolio Polish** | 15–18 | dashboards | API + dashboard + writeup | p95 latency target; public case study |
| **P9 Agentic (stretch)** | 18–20 | LangGraph | Data-discovery + research agents | Optional |

**Dependencies (critical path):** P1→P2→P3→P4→P5. P3 (entity resolution) is the single
hardest dependency — everything downstream multiplies its error. P5 is a **hard kill-gate**:
if SAI shows no forward predictive power, re-scope P6–P8 toward "honest negative result +
methodology" (still a strong portfolio piece) rather than building forecasters on noise.

**Top risks:** (1) scraper maintenance eating the solo budget; (2) legal/ToS exposure from
aggressive crawling; (3) look-ahead bias silently inflating backtests; (4) small effective
sample (daily × few hundred tickers) → fragile statistics; (5) scope sprawl across 20+ sources.

**Deliverables:** the proprietary dataset (the real product), the SAI, a validated/invalidated
signal report, a forecasting model, a dashboard, and a written research case study.

---

## STEP 2 — Data Acquisition Master Plan

Grouped by tier. For each: **why / signals / collection / storage / refresh / volume / issues.**

### Tier A — Reliable backbone (build first, low risk)

**Reddit (PRAW + archives)**
- *Why:* richest retail-attention signal; WSB/stocks lead price chatter.
- *Signals:* mention counts, upvotes/comments (engagement), sentiment, ticker co-mentions, post velocity.
- *Collection:* PRAW for live (~60 req/min); Kaggle/Academic archives for historical backfill
  (Pushshift largely gated now — use existing Kaggle dumps).
- *Storage:* raw JSON → bronze parquet partitioned by `dt/subreddit`; parsed → Postgres `documents`.
- *Refresh:* hourly for hot subs; daily rollups.
- *Volume:* ~50–200k posts+comments/day → ~1–3 GB/month raw.
- *Issues:* API auth, deleted content, bot/pump spam, Pushshift historical gaps.

**News RSS (feedparser)**
- *Why:* institutional-attention timing; headlines often precede price reaction intraday.
- *Signals:* headline sentiment (FinBERT), publication cadence, entity mention bursts.
- *Collection:* poll curated feeds (WSJ/CNBC/MarketWatch/Reuters); dedup by URL+title hash.
- *Storage:* bronze parquet by `dt/feed`; Postgres `documents`.
- *Refresh:* every 15–30 min.
- *Volume:* ~1–5k items/day, <100 MB/month.
- *Issues:* RSS truncation (summary only), paywalls, feed instability.

**Yahoo Finance (yfinance)**
- *Why:* the **label source** — OHLCV, returns, earnings dates. Without it there's nothing to validate against.
- *Signals:* forward returns (targets), volume, realized vol, earnings calendar.
- *Collection:* daily pull for universe; treat earnings/fundamentals as *point-in-time* (avoid restated-data leakage).
- *Storage:* Postgres `market_data` + parquet snapshot.
- *Refresh:* daily after close.
- *Volume:* tiny, <10 MB/month.
- *Issues:* unofficial API (breaks without notice — isolate behind collector), survivorship bias, splits/adjustments.

**Kaggle datasets**
- *Why:* instant historical depth (years of Reddit/news/reviews) without scraping — backfill shortcut.
- *Signals:* historical mentions/sentiment/reviews for backtesting.
- *Collection:* `kaggle` API; catalog + license-check each dataset.
- *Storage:* raw archive in object store; normalized into bronze.
- *Refresh:* one-time / occasional.
- *Volume:* GBs, bursty.
- *Issues:* licensing varies, schema drift, unknown collection methodology (provenance risk).

### Tier B — High-value, moderate effort

- **Hacker News (Firebase API)** — tech/product attention leading indicator; clean free API;
  signals = points/comments velocity on company/product mentions. Hourly. Best ROI alt-source.
- **GitHub Trends (REST API)** — developer adoption proxy for dev-tool/infra companies;
  signals = star velocity, repo creation around a product. Daily. Rate-limited but clean.
- **Product Hunt (API/GraphQL)** — early consumer-product attention; signals = upvotes/launch velocity.
  Daily. Small volume; API access gating is the main friction.
- **Google Trends alternatives** — search-attention proxy. Official Trends API is unstable; use
  `pytrends` cautiously or substitute **Wikipedia pageview API** (clean, official, underrated)
  and free SERP-trend datasets. Daily/weekly.

### Tier C — Web scraping (use surgically, not broadly)

- **BeautifulSoup + requests** — static pages. Cheapest, brittle to layout changes.
- **Playwright** (preferred over Selenium) — JS-heavy pages, infinite scroll; better async + auto-wait. Higher maintenance.
- **Scrapy** — only if you need scale/pipelines on one stable target; overkill for exploration.

*Why limited:* for a solo dev, scrapers are a **maintenance tax that compounds**. Strategy:
scrape ≤2 high-signal targets, prefer APIs/datasets everywhere else.
- *Storage:* raw HTML snapshot (reproducibility) → parsed bronze.
- *Issues:* ToS/legal, IP blocks, CAPTCHAs, layout drift, robots.txt compliance.

### Tier D — AI-powered crawling (evaluate, don't depend on)

- **Crawl4AI / Firecrawl / browser-use** — LLM-assisted extraction; great for *one-off discovery*
  of new sources and messy pages where writing selectors isn't worth it.
- *Why:* dramatically lowers cost of *exploring* a new source; resilient to layout changes.
- *Issues:* **per-page LLM cost** (can balloon), latency, non-determinism, hallucinated extractions
  → must validate. **Verdict:** use for **data discovery and prototyping**, not high-volume production.

### Tier E — Public/government data

- **Wikipedia pageviews, SEC EDGAR (filings/8-K timing), BLS/Census/data.gov** — slow-moving but
  **clean, free, legally safe, high-provenance**. EDGAR 8-K timestamps are a great *event anchor*
  for "did attention precede the disclosure?" Low maintenance. Underused by retail projects → differentiator.

### Source Comparison Scorecard (1=poor, 5=excellent)

| Source | Cost | Scalability | Reliability | Legal safety | Freshness | Maint. effort | Signal quality |
|---|---|---|---|---|---|---|---|
| Reddit API | 5 | 3 | 4 | 4 | 5 | 3 | 5 |
| News RSS | 5 | 4 | 4 | 5 | 5 | 4 | 4 |
| Yahoo Finance | 5 | 4 | 3 | 4 | 5 | 3 | 5 (labels) |
| Kaggle | 5 | 5 | 5 | 3 | 1 | 5 | 4 |
| Hacker News | 5 | 5 | 5 | 5 | 5 | 5 | 4 |
| GitHub trends | 5 | 4 | 5 | 5 | 4 | 4 | 3 |
| Product Hunt | 4 | 4 | 4 | 4 | 4 | 4 | 3 |
| Wikipedia views | 5 | 5 | 5 | 5 | 4 | 5 | 3 |
| SEC EDGAR | 5 | 5 | 5 | 5 | 3 | 4 | 4 (anchor) |
| Web scraping | 4 | 2 | 2 | 2 | 4 | 1 | 3 |
| AI crawling | 2 | 2 | 3 | 3 | 4 | 3 | 3 |

**Read:** lean hard on Tier A+B (high reliability/legal/low-maintenance). Scraping and AI-crawling
score low on the axes that hurt a solo dev most (maintenance, legal). Use them for *discovery and
a couple of surgical targets*, never as the backbone.

---

## STEP 3 — Data Lake Architecture (Medallion)

Solo-dev-pragmatic: **local object store (MinIO/parquet) + Postgres + pgvector**, not a cloud
warehouse. Same layered logic that scales later.

```
data/
├── 00_raw/        (Bronze) immutable, source-shaped, append-only
│   ├── reddit/dt=2026-06-22/subreddit=wallstreetbets/*.json.gz
│   ├── rss/dt=.../feed=cnbc/*.json
│   ├── yahoo/dt=.../*.parquet
│   ├── hackernews/ github/ producthunt/ ...
│   └── _manifests/   ingestion-run metadata (source, rows, hash, ts)
├── 10_processed/  (Silver) cleaned, deduped, entity-linked, typed parquet
│   ├── documents/dt=...           one row per normalized doc
│   ├── document_entities/         doc↔ticker links + confidence
│   └── market/                    adjusted OHLCV, earnings calendar
├── 20_features/   (Gold) point-in-time feature panels for modeling
│   ├── sai_components/entity=.../  mention/sentiment/topic/engagement
│   ├── sai_daily/                  composite index
│   └── training_panels/            entity×date×features×forward_return
├── 30_vector/     embeddings (pgvector primary; FAISS export optional)
└── 40_analytics/  (Gold/serving) curated marts for dashboards/API
    ├── signal_leaderboard/  forecasts/  backtest_results/
```

**Layer decisions & why:**
- **Raw (Bronze):** immutable + append-only + raw HTML/JSON kept. Non-negotiable for research —
  must be able to re-derive everything and prove provenance. Partition by `dt/source`.
  Store **`ingested_at`** with every record (anti-look-ahead).
- **Processed (Silver):** dedup (content hash), language filter, entity resolution, typed columns.
  Parquet for columnar analytics; the *queryable* slice mirrors into Postgres `documents`/`document_entities`.
- **Feature (Gold):** the crown jewel — **point-in-time** panels where every feature is "as known
  at date D," joined to **forward** returns as labels. This is where look-ahead bias lives or dies.
  Stored both as parquet (training) and Postgres `sai_daily` (serving).
- **Vector:** embeddings in **pgvector** keyed to `document_id`; one DB to operate. FAISS only as an
  offline export if you need batch ANN at scale.
- **Analytics:** denormalized marts the API/dashboard read — fast, no heavy joins at request time.

**Postgres tables:** `entities, sources, documents, document_entities, sentiment_scores, topics,
document_topics, embeddings(vector), sai_daily, market_data, earnings_calendar, forecasts, reports,
ingestion_runs, data_quality_checks, model_registry`. Every fact-table row carries `published_at`
(event) **and** `ingested_at`/`computed_at` (known) timestamps.

**Why a lake + warehouse hybrid:** parquet lake = cheap, reproducible, ML-friendly; Postgres =
transactional serving + relational integrity. Research reproducibility *and* a fast API without a cloud bill.

---

## STEP 4 — Data Quality Framework

Bad alt-data doesn't just add noise — it adds *correlated* noise (coordinated pumps look exactly like
real signal). DQ is a first-class pipeline stage with a `data_quality_checks` table logging every assertion.

| Threat | Detection approach |
|---|---|
| **Duplicates** | Exact: SHA-256 `content_hash` unique constraint. Near-dup: MinHash/SimHash + cosine on embeddings > threshold → collapse. Cross-source dedup (syndicated news). |
| **Spam / pump posts** | Rule layer (URL density, repeated cashtag stuffing, copypasta templates) + classifier on text features; flag sudden single-account ticker floods. |
| **Bots** | Account-age, post-frequency, burstiness, duplicate-text-across-accounts, account-creation clustering. Score → `author_bot_score`; down-weight in SAI rather than hard-delete. |
| **Fake engagement** | Statistical outlier detection on upvote/comment ratios vs. subreddit baseline; vote-velocity spikes inconsistent with comment growth; coordinated timing. |
| **Low-quality sources** | Per-source **signal-quality score** = backtested forward-IC contribution + reliability uptime + spam rate. Auto-demote decaying sources. |
| **Data drift** | Population Stability Index (PSI) / KL-divergence on feature distributions vs. trailing baseline; volume anomaly detection per source; embedding-centroid drift; alert on schema/freshness breaks. |

**Principles:** (1) *quarantine, don't delete* — move suspect rows to a `_quarantine` partition so
research stays reproducible; (2) **weight, don't filter** where possible (a bot-heavy mention still
carries information about coordination); (3) every check writes a pass/fail row with metrics → DQ is
a monitored dashboard, not a one-off script; (4) **freshness alerts** (a silently stalled collector
degrades every signal — alert on staleness, not just errors).

---

## STEP 5 — Research Plan (Lab Mode)

Run this like a lab: each experiment is a falsifiable hypothesis with a pre-registered success metric
and a **reserved holdout** (never touched during iteration — guards against p-hacking). Core metric:
**Information Coefficient (IC)** = rank-corr(signal, forward return), plus decile-spread returns net of costs.

Format: **ID · Hypothesis · X→Y · Method · Success metric.**

**Attention → Price**
1. Reddit mention *growth* predicts next-5d return. → lead-lag rank regression. → IC>0.03, p<0.05.
2. Mention **acceleration** (2nd derivative) beats raw mention level. → compare IC of level vs Δ vs Δ². → accel IC > level IC.
3. Cross-source mention *agreement* (Reddit+HN+News align) predicts larger/longer moves. → conditional IC by agreement count.
4. Unusual mention volume vs. own trailing baseline (z-score) predicts volatility expansion. → IC vs realized vol.
5. Ticker **co-mention networks** (A discussed with B) predict spillover returns. → graph-propagated signal IC.

**Sentiment → Fundamentals**
6. Sentiment *momentum* predicts earnings surprise sign. → classification AUC vs actual EPS surprise.
7. Sentiment momentum predicts **revenue** surprise (consumer names). → AUC on revenue beats.
8. Sentiment **dispersion** (disagreement) predicts post-earnings volatility. → IC vs |earnings-day return|.
9. News sentiment leads Reddit sentiment (or vice versa) — who's the leader? → cross-correlation lag analysis.
10. Negative-sentiment bursts predict drawdowns better than positive predict rallies (asymmetry). → conditional IC by sign.

**Topic / Trend → Adoption**
11. Topic **velocity** (BERTopic cluster growth) predicts consumer-adoption proxies (review counts, app ranks). → lead-lag IC.
12. Emergence of a *new* topic cluster around a company predicts abnormal returns within N days. → event study CAR.
13. Topic velocity on a *product* predicts the parent company's next-quarter revenue surprise. → AUC.
14. GitHub star velocity predicts dev-tool company attention/returns. → IC for software universe.
15. Wikipedia pageview spikes lead price for consumer brands. → lead-lag IC.

**Microstructure / Anchoring / Robustness**
16. Attention shifts lead SEC 8-K disclosures (does the crowd know first?). → time-to-event distribution vs filing.
17. Engagement growth adds orthogonal signal beyond mention growth (incremental IC). → multivariate IC.
18. Signal decay: at what horizon (1/3/5/10/20d) is SAI IC maximal? → IC-by-horizon curve.
19. Does the signal survive transaction costs and realistic turnover? → net Sharpe of decile long-short.
20. Regime dependence: does SAI work in high-vol vs low-vol regimes? → IC conditioned on VIX bucket.
21. Bot-filtered SAI outperforms raw SAI (does DQ add alpha?). → IC raw vs cleaned.
22. Small-cap vs large-cap: where is social signal strongest? → IC by market-cap quintile.
23. *Negative control:* does SAI "predict" a **lagged/random** return (it shouldn't)? → IC≈0 confirms no leakage.

**Each experiment outputs:** a notebook + a one-page result card (hypothesis, data window, IC/AUC,
plot, verdict, caveats). The collection of result cards *is* your portfolio's research credibility.

---

## STEP 6 — Agent Architecture

Two-speed design: a **deterministic pipeline** (reproducible signals) plus an **agent layer** for the
open-ended data-discovery/research work where LLM judgment genuinely helps. **Hard rule: agents
discover, evaluate, narrate, and route — they never compute the signal numbers.** Built on LangGraph
in P9; the specs below define the tool boundaries now.

| Agent | Inputs | Outputs | Tools | Responsibilities | Success metric |
|---|---|---|---|---|---|
| **Data Discovery** | domain/topic, coverage gaps | candidate source list + access notes | WebSearch, Firecrawl, API catalogs | Find new datasets/APIs/feeds worth ingesting | # viable new sources/week; % passing eval |
| **Dataset Evaluation** | candidate source | scorecard (cost/legal/freshness/signal) + go/no-go | schema profiler, license parser, sample fetch | Score sources on the 7 axes; flag legal risk | Eval accuracy vs later realized value |
| **Crawler** | target URL/site, extraction schema | structured records + raw snapshot | Crawl4AI/Firecrawl, Playwright | AI-assisted extraction from messy/new pages | extraction precision; cost/page |
| **Scraper** | stable target + selectors | parsed records | Playwright/BS4/Scrapy | Deterministic high-volume pulls from known sites | uptime; rows/day; breakage rate |
| **Data Cleaning** | raw documents | deduped, entity-linked, DQ-scored rows | dedup/MinHash, entity resolver, bot scorer | Enforce Step-4 quality gates | dup rate <2%; entity precision ≥90% |
| **Topic Discovery** | document embeddings | topic clusters + velocity series | sentence-transformers, BERTopic, HDBSCAN | Surface emerging narratives | topic coherence (c_v); stability |
| **Sentiment** | documents | sentiment/momentum scores | FinBERT, local LLM, calibration | Score tone; compute momentum | F1 vs labeled set; calibration |
| **Forecast** | feature panel | return/surprise predictions + intervals | XGBoost/LightGBM (→TFT) | Train/serve forecasts, no leakage | OOS IC; beats naive baseline |
| **Research** | hypothesis + data | experiment result card | backtest harness, stats, plotting | Run Step-5 experiments rigorously | reproducibility; pre-registered metric met |
| **Report** | signals + results | grounded NL report | Claude (hybrid), templates, number-injection | Narrate findings; **cite DB numbers only** | 0 hallucinated figures; reviewer rating |

**Orchestration (P9):** a Supervisor routes Discovery→Evaluation→(Crawler/Scraper)→Cleaning into the
lake, then Research/Forecast/Report on demand, with LangGraph checkpoints for auditability and
human-in-the-loop approval before a new source goes live.

---

## STEP 7 — 12-Week Milestone Plan (week by week)

Weeks 1–8 ≈ data (≈70%); Weeks 9–12 ≈ modeling/reasoning/serving (≈30%).

**W1 — Source recon & legal.** *Obj:* prove top sources. *Tasks:* spike collectors for Reddit/HN/RSS/Yahoo;
license & ToS log; robots.txt review. *Research:* draft 23 experiments + holdout split policy.
*Deliverable:* Source Scorecard + legal register.

**W2 — Backfill via Kaggle + market labels.** *Obj:* historical depth fast. *Tasks:* ingest 2–3 Reddit/news
Kaggle dumps; Yahoo OHLCV + earnings calendar for universe. *Research:* exploratory mention/return
correlation (sanity, train only). *Deliverable:* bronze raw lake + `market_data`.

**W3 — Ingestion backbone.** *Obj:* productionize collectors on the `Collector` ABC. *Tasks:*
Reddit+RSS+Yahoo+HN live, idempotent, `ingestion_runs` logging, scheduler. *Deliverable:* 30-day
continuous feed, dedup working.

**W4 — Entity resolution.** *Obj:* text→ticker mapping. *Tasks:* alias/cashtag dictionary, disambiguation,
confidence scores; labeled eval set. *Deliverable:* `document_entities` at ≥90% precision.

**W5 — Data Quality framework.** *Obj:* trust the data. *Tasks:* near-dup (MinHash), bot/spam scorers,
drift (PSI), DQ dashboard + quarantine. *Deliverable:* `data_quality_checks` live; quality report.

**W6 — Enrichment: sentiment.** *Obj:* tone signal. *Tasks:* FinBERT scoring service, momentum
computation, calibration vs labeled set. *Research:* Exp 6–10. *Deliverable:* `sentiment_scores` + result cards.

**W7 — Enrichment: embeddings + topics.** *Obj:* narrative signal. *Tasks:* sentence-transformer
embeddings→pgvector, BERTopic, topic-velocity series. *Research:* Exp 11–15. *Deliverable:* `topics`,
`document_topics`, vector layer.

**W8 — Feature store (point-in-time).** *Obj:* leak-free panels. *Tasks:* build `training_panels`
(entity×date×features×forward_return) with strict as-of joins; negative-control feature.
*Deliverable:* reproducible feature build + leakage test (Exp 23).

**W9 — SAI + backtest harness.** *Obj:* the index + validation. *Tasks:* 4 sub-signals→composite;
walk-forward IC, decay, decile spread, costs. *Research:* Exp 1–5, 17–22.
*Deliverable:* **SAI + signal-validation report (KILL-GATE).**

**W10 — Forecasting.** *Obj:* predictive model. *Tasks:* XGBoost/LightGBM on panels, temporal CV,
model registry, beat naive baseline. *Deliverable:* `forecasts` + metric cards.

**W11 — Reasoning & reporting.** *Obj:* grounded narratives. *Tasks:* hybrid LLM reports (numbers
injected from DB), grounding eval. *Deliverable:* auto-report with 0 hallucinated figures.

**W12 — Serving & portfolio polish.** *Obj:* show it. *Tasks:* FastAPI signal/forecast/search endpoints,
a Streamlit/dashboard, write the case study (incl. honest negatives). *Deliverable:* running demo + public writeup.

---

## STEP 8 — Critical Evaluation & Final Execution Strategy

**Weaknesses**
- **Small-sample statistics.** Daily data × a few hundred tickers over ~2–3 years is a *thin* panel;
  IC estimates will be noisy and easy to fool yourself with. → *Fix:* pool across entities, use
  rank-based stats, bootstrap confidence intervals, and **reserve a true holdout** touched only once.
- **Look-ahead bias is the silent killer.** yfinance restated fundamentals, earnings dates, and
  "ingested vs published" sloppiness will inflate backtests. → *Fix:* point-in-time everything; the
  negative-control experiment (Exp 23) is mandatory.
- **Source breadth is a trap.** 20+ sources is a *research menu*, not a build list. → *Fix:* productionize
  ~5 (Reddit, RSS, Yahoo, HN, one Kaggle backfill); treat the rest as time-boxed discovery spikes.
- **Scraper/AI-crawl maintenance** compounds and will quietly consume your week. → *Fix:* APIs/datasets
  first; ≤2 surgical scrapers; AI-crawling only for discovery, with a hard monthly $ cap.

**Bottlenecks**
- **Entity resolution (W4)** gates all signal quality — under-resourcing it poisons everything. Budget buffer.
- **Compute for embeddings/topics** on a backfill can be slow on a laptop. → batch offline, cache,
  consider a one-time GPU rental.
- **You (single point of failure).** Context-switching across data+ML+infra+writing is the real constraint.

**Failure points**
- The kill-gate fails (no signal). → **Reframe, don't abandon:** "rigorous investigation showing social
  signal is weak/regime-dependent, with leak-free methodology" is a *more credible* portfolio piece than
  another over-fit demo. Plan for this outcome explicitly.
- Legal/ToS complaint from a scraped site. → respect robots.txt, rate-limit, prefer official APIs, keep the legal register.
- Coordinated-manipulation data poisoning your signal. → the bot/fake-engagement DQ layer is a *feature to showcase*.

**Improvements to the plan**
1. **Add SEC EDGAR 8-K timestamps as an event anchor** early — clean "did attention lead disclosure?" experiments, high provenance.
2. **Wikipedia pageviews + HN** are underrated, legally safe, low-maintenance — promote above scraping.
3. **Treat result cards as the deliverable.** The portfolio differentiator is documented, falsifiable research with honest negatives — not a dashboard.
4. **Version the dataset** (DVC or dated parquet snapshots) so every experiment is reproducible — this *is* the moat.

**Final execution strategy (solo, portfolio-grade)**
> Build a **narrow, deep, leak-free panel from ~5 reliable sources**, instrument it with real
> **data-quality + point-in-time** discipline, and run it like a **lab** that pre-registers ~20
> falsifiable experiments against a reserved holdout. Hit the **signal-validation kill-gate at W9
> before** investing in modeling. Keep scraping/AI-crawling as bounded discovery spikes, not
> infrastructure. Ship a **running demo + a written research case study that reports honest results**
> (positive *or* negative). The dataset and the rigor are the product; the model is the epilogue.
