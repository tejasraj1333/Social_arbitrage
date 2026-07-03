# Entity Resolution & Data Quality — Phase 3

How SAM links documents to tradable entities (`document_entities`) and keeps
the dataset trustworthy (`data_quality_checks`). This phase is architecture
milestone **M2** — the single hardest dependency on the critical path:
everything downstream (sentiment, SAI, backtests) multiplies its error.

## Where it sits

```
ingest (P2) ──► resolve (P3) ──► dq (P3) ──► nlp / signals (P4+)
   documents      document_entities   data_quality_checks
```

The `processing` package never imports `ingestion`; the stages communicate
only through the database. Run order: `sam ingest` → `sam resolve` → `sam dq`
(each is idempotent and cron-safe).

## Resolution method

Deterministic, rule-based matching (`sam.processing.resolver`) — no models,
no LLMs, so signals rebuild reproducibly. Per entity, the strongest matching
rule wins:

| Rule | Example | Confidence | Notes |
|---|---|---|---|
| cashtag | `$NVDA` | 1.00 | case-insensitive; `$` sigil disambiguates |
| bare ticker | `NVDA` | 0.90 | case-sensitive; skipped for `AMBIGUOUS_TICKERS` (common English words like `ALL`, `IT`, `NOW`) |
| alias / name | `Nvidia`, `Amazon.com, Inc.` | 0.80 | case-insensitive, word-boundary lookarounds (handles names ending in punctuation) |

Confidence is a **weight, not a filter** — downstream signals down-weight
low-confidence links instead of dropping them (quarantine-don't-delete).

The dictionary is curated in `config/sources.yaml` (`universe[].aliases`) and
synced to the `entities` table with `sam seed --update`. After changing
aliases, run `sam resolve --all` to re-scan; links refresh via
`ON CONFLICT DO UPDATE` and `resolved_at` records when each link became
*known* (point-in-time rule).

Incrementality: every scanned document — matched or not — gets a
`documents.resolved_at` stamp, so a normal `sam resolve` only touches new
documents. The pipeline commits per batch; an interrupted run loses at most
one batch and never stamps a document without its links.

## Evaluation (the ≥90% precision gate)

`data/eval/entity_labels.jsonl` holds **54 real ingested headlines**
(11 positive / 43 negative) hand-labeled with the universe tickers they truly
mention. The negatives include genuinely adversarial cases — e.g.
*"**Mamdani**-backed candidates…"* (contains the substring "amd") and
tech-adjacent noise (Meta, Google, Samsung, SK Hynix, DeepSeek).

Measured on 2026-07-03 (`sam resolve --evaluate`):

| Metric | Value | Gate |
|---|---|---|
| Precision | **1.000** | ≥ 0.90 ✅ |
| Recall | **1.000** | ≥ 0.80 (sanity floor) ✅ |
| Examples | 54 (11 pos / 43 neg) | ≥ 50 |

The gate is encoded as a permanent pytest
(`tests/test_evaluate.py::test_resolver_precision_gate`), so any rule or
dictionary change that drops precision below 0.90 fails CI. Evaluation is
DB-free (dictionary built from config), so it runs everywhere.

Live validation on the real corpus (256 documents): 13 links produced, all
manually verified correct — including two matched via RSS summary text rather
than the headline ("Nvidia was down 16%" inside a DeepSeek-rout story).
A second `sam resolve` scanned 0 documents (watermark idempotency).

## Data-quality framework

`sam dq` runs every check and appends pass/warn/fail rows to
`data_quality_checks` — quality is a monitored time series, not a one-off
script. Evidence (offending ids) goes into `details` (JSONB).

| Check | Scope | Thresholds | Live result (2026-07-03) |
|---|---|---|---|
| `duplicate_rate` | last ≤1000 docs | warn >1%, **fail >2% (P3 gate)** | **0.39% pass** — caught the real cross-feed pair ("start-up" vs "startup" syndication) |
| `freshness` | per source | warn >26h, fail >48h (or no success ever) | ~16.7h, all pass |
| `volume_anomaly` | per source | warn <0.5× trailing mean, fail on 0 | pass (insufficient history noted honestly) |
| `resolution_coverage` | global | warn only when 0 links across ≥100 docs (dictionary rot) | 5.08% — expected for a 6-ticker universe vs. general market news |

Near-dup detection: token-set Jaccard ≥ 0.85 on titles (intra-word hyphens
collapsed — the variation actually observed in the wild), O(n²) with a size
prefilter. Exact duplicates cannot exist (`content_hash` UNIQUE).

## Known limitations (deliberate, documented)

- **Generic-word aliases** ("Apple", "Amazon") will false-positive on
  non-company prose ("apple pie", "Amazon rainforest"). Market-news feeds
  essentially never use them that way — the measured sample shows zero such
  FPs — but Reddit-scale casual text will need context disambiguation
  (planned in the NLP phase, P4). `tests/test_evaluate.py` encodes one such
  case as expected-FP in its metric-math fixture to keep the weakness visible.
- **No fuzzy matching** (misspellings like "Nvidea" are missed) — precision
  is the gated metric; recall costs are accepted for now.
- **Bot/spam scoring** (blueprint W5) requires author-level Reddit data and
  joins the DQ framework once Reddit credentials land.
- **MinHash** replaces the O(n²) near-dup scan when Reddit volume arrives.

## Operations

```bash
SAM_DB__PORT=5433 uv run sam seed --update   # after editing aliases in config
SAM_DB__PORT=5433 uv run sam ingest          # collect (cron/Task Scheduler)
SAM_DB__PORT=5433 uv run sam resolve         # link new documents
SAM_DB__PORT=5433 uv run sam dq              # record quality checks (exit 1 on fail)
uv run sam resolve --evaluate                # precision gate (no DB needed)
```
