# SAI Methodology — Phase 5

How SAM turns resolved + enriched documents into the **Social Arbitrage
Index**: a daily, per-entity composite of four sub-signals persisted in
`sai_daily`. This phase is architecture milestone **M4**; its gate is
**deterministic rebuild from raw** — `sam sai --rebuild` must reproduce the
panel value-for-value. The panel is the direct input to P6 signal validation
(the forward-IC kill-gate), so every definition below optimizes for
point-in-time correctness over cleverness.

## Where it sits

```
ingest (P2) ─► resolve (P3) ─► enrich (P4) ─► topics (P4) ─► sai (P5) ─► validation (P6)
  documents     document_entities  sentiment_scores  topics/…     sai_daily     (kill-gate)
```

`sam.signals` imports only `sam.core` and `sam.storage`; stages communicate
exclusively through the database. Daily run order matters:
`sam ingest → sam resolve → sam enrich → sam topics → sam sai → sam dq` —
SAI must see the day's links and scores (a gap heals with `--rebuild`, but
the chain order avoids creating one).

## Point-in-time rules (the load-bearing decisions)

1. **Documents bucket by the UTC day of `ingested_at`** — the *known* time,
   never `published_at`. Closed days are immutable (a document can only be
   ingested "now", never backdated), which is exactly what makes rebuilds
   deterministic and backtests leak-free. This is the architecture's
   "training and backtests join on known time only" rule applied to signals.
2. **Staleness guard**: documents whose `published_at` predates ingestion by
   more than `signals.max_doc_age_days` (default 7) are excluded from *all*
   aggregates — a Kaggle backfill landing today is history arriving late, not
   an attention spike today. Documents without `published_at` are treated as
   fresh (staleness can't be proven).
3. **Topic version as-of**: topic velocity for day D uses the latest
   `topic_model_version` whose fit predates the end of D. Versions are
   append-only, so a rebuild of a past day selects the same version that a
   live run selected then.
4. **Sentiment model pinning**: only rows written by the *configured*
   `nlp.sentiment_model` are read. Changing that model (or any `signals.*`
   setting) changes signal values ⇒ run `sam sai --rebuild` afterwards.
5. **Only closed UTC days are computed** (through yesterday). A partial day
   would change on re-run; closed days never do.

## Sub-signals

All aggregates are per (entity, UTC day) over *fresh, linked* documents,
weighted by `document_entities.confidence` — weight, don't filter.

| Component | Daily raw aggregate | Transform |
|---|---|---|
| `mention_growth` | Σ link confidence | growth vs trailing mean |
| `sentiment_momentum` | confidence-weighted mean of signed sentiment (+score / −score / 0 for positive/negative/neutral), over scored docs only | difference vs trailing mean of *defined* days |
| `topic_velocity` | per topic: probability-weighted doc count (global) | entity's (confidence × probability)-weighted mean of its day-D topics' growth rates |
| `engagement_growth` | Σ confidence × (engagement `score` + `comments`) | growth vs trailing mean |

**Growth transform** (`sam/signals/compute.py:growth`):
`(value_D − mean(trailing window)) / max(mean, 1.0)` over a
`signals.window_days` (default 7) window, missing days = 0 (a dense panel:
no activity *is* an observation). The `max(mean, 1.0)` floor keeps
small-count growth bounded and defined against a zero baseline.
**Momentum** differs in that missing days are *gaps* (sentiment on an
unscored day is unknown, not zero).

**History gate**: every component is NULL until the panel is
`signals.min_history_days` (default 3) old — a growth rate against no
baseline is noise, and NULL ("insufficient history") is different
information than 0.0 ("no change"). The panel origin is the first day with
any fresh linked document.

## Composite

Per day, each component is converted to **cross-sectional centered ranks**
in [−1, 1] (average ranks for ties); `sai_score` is the weighted mean of the
entity's available component ranks with weights `signals.weight_*`
(default 0.25 each) — absent components drop out and their weight
renormalizes away. `sai_rank` orders scores within the day (1 = strongest,
ties broken by entity id).

Why ranks rather than the blueprint's trailing z-scores: the growth
transforms already normalize each series against its own trailing window;
a second time-series normalization would need a std estimate (~10+ days of
history the young panel doesn't have), while ranks are scale-free, robust at
small n, and the validation metric this feeds (Information Coefficient) is
rank correlation anyway — nothing the kill-gate measures is lost. Revisit
when the entity universe grows past ~30 names (documented deviation).

Zero-activity days produce real rows: an entity going from 5 mentions/day to
0 is an attention collapse (mention_growth −1.0), not missing data.

## Determinism (the phase gate)

`build_panel` is a pure function: repository rows in, panel out — no clock,
no config reads, no sampling. Stable sorts everywhere (ties break on ids),
Python-side day bucketing (no SQL dialect drift), and `computed_at` is the
only non-deterministic column (excluded from rebuild comparisons). Proven
two ways: a pipeline test asserts a rebuild reproduces byte-identical values
on SQLite, and the live gate checksums `sai_daily` before/after
`sam sai --rebuild` on Postgres.

## Operations

```bash
SAM_DB__PORT=5433 uv run sam sai              # incremental: new closed days only
SAM_DB__PORT=5433 uv run sam sai --rebuild    # after config/model changes
SAM_DB__PORT=5433 uv run sam dq               # includes sai_freshness (warns >=2 days behind)
```

The pipeline skips honestly (exit 0, warning log) when there are no linked
documents or no closed days yet. `sai_freshness` in `sam dq` warns when the
panel trails by ≥2 closed days, or is missing despite ≥100 linked documents.

## Known limitations (deliberate, documented)

- **Young panel = NULL components.** With `min_history_days=3`, real values
  appear on the panel's 4th day and stabilize as the 7-day windows fill.
  Honest NULLs now beat fabricated growth rates forever.
- **Cross-sectional ranks over 6 entities are coarse** (score granularity
  ~0.4). Fine for machinery validation; the composite sharpens automatically
  as the universe grows.
- **Engagement is HN-only today** (RSS carries no counts); the component is
  effectively "HN engagement" until Reddit lands. Reddit's collector must
  normalize onto the `score`/`comments` engagement keys.
- **Enrichment lag freezes into closed days** if the chain runs out of
  order: a day computed before its documents were scored keeps NULL
  momentum until a `--rebuild`. Chain order + the `enrichment_coverage` and
  `sai_freshness` DQ checks guard this operationally.
- **FinBERT's up-is-good bias** (documented in `docs/nlp_enrichment.md`)
  propagates into `sentiment_momentum` unmitigated — mitigation is a P6
  concern, evaluated against measured IC, not guessed at here.
