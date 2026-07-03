# NLP Enrichment & Feature Store — Phase 4

How SAM turns ingested documents into model-ready features: finance-domain
sentiment (`sentiment_scores`), semantic embeddings (`embeddings`, pgvector)
and versioned topics (`topics` / `document_topics`). This phase is
architecture milestone **M3**; its outputs are the direct inputs to the SAI
composite (P5), so reproducibility and point-in-time integrity dominate the
design.

## Where it sits

```
ingest (P2) ─► resolve (P3) ─► enrich (P4) ─► topics (P4) ─► SAI (P5)
  documents     document_entities   sentiment_scores      sai_daily
                                    embeddings
                                    topics / document_topics
```

`sam.nlp` never imports `sam.ingestion` or `sam.processing` — stages
communicate only through the database. `resolve` and `enrich` are fully
decoupled (independent watermarks, either order); `topics` requires `enrich`
first because it clusters the *stored* embeddings.

Run order: `sam ingest` → `sam resolve` → `sam enrich` → `sam topics` →
`sam dq` (each idempotent and cron-safe; `topics` is meant for a slower
cadence, e.g. weekly, since each run appends a new topic-model version).

## Models & abstraction

| Role | Default model | Output | Config key |
|---|---|---|---|
| Sentiment | `ProsusAI/finbert` | label ∈ {positive, negative, neutral} + confidence | `nlp.sentiment_model` |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` | 384-dim unit-norm vector | `nlp.embedding_model` |
| Topics | BERTopic (seeded UMAP + HDBSCAN, stopword-free c-TF-IDF) | versioned topic clusters | — |

Design rules (`sam/nlp/models.py`):

- **Lazy heavy imports** — torch/transformers load on first inference, so the
  core package, CLI startup, and CI run without the `nlp` extra. Unit tests
  inject protocol-conforming fakes; a subprocess test enforces that importing
  `sam.nlp` pulls no heavy library.
- **Model id on every row** — outputs from different models never mix
  silently; PK is `(document_id, model)`, so a model upgrade writes new rows
  while the old model's rows survive for comparison.
- **Deterministic inference** — no sampling anywhere; UMAP is seeded
  (`random_state=42`). Re-running `sam enrich --all` rewrites identical
  values; re-running `sam topics` on an unchanged corpus reproduces identical
  clusters (verified live: two consecutive fits → identical 2 topics /
  31 outliers / 229 assignments). This is the P4 "features rebuild
  deterministically" gate.
- Embedding width is **schema**, not config: `vector(384)` (pgvector is fixed
  width). A different-width model requires a migration; the pipeline
  fail-fasts on mismatch (`EnrichmentError`) before writing anything.

Device is `cpu` by default (`SAM_NLP__DEVICE=cuda` to opt in — both models fit
small GPUs). Throughput measured live: 260 documents scored + embedded in
~60 s on CPU, including model load.

## Incremental enrichment

`documents.enriched_at` mirrors the P3 `resolved_at` watermark: NULL = not
yet scanned; stamped even when a document has no usable text, so a normal
`sam enrich` touches only new documents (verified live: second run scans 0).
Per-batch commits mean an interrupted run loses at most one batch and never
stamps a document without its rows. `sam enrich --all` re-scans after a model
change; rows refresh via `ON CONFLICT DO UPDATE`.

The text scored is `title + "\n" + raw_text` — exactly the fields the entity
resolver matches on, so sentiment describes the same text that produced the
entity links. **Entity-aware sentiment is a join, not a second model**:
downstream (P5) weights document sentiment by `document_entities.confidence`
per linked entity. Aspect-level sentiment (different tone for two tickers in
one document) is deliberately deferred until evidence demands it.

## Sentiment evaluation (the ≥0.70 macro-F1 gate)

`data/eval/sentiment_labels.jsonl` holds **63 real ingested headlines**
(11 positive / 27 negative / 25 neutral) hand-labeled under a written policy:
label by the implied market/business direction for the headline's subject;
`neutral` for purely factual, procedural, or balanced items. The gate
(**macro-F1 ≥ 0.70**) was pre-registered before the first measurement — same
no-p-hacking discipline as the P3 precision gate.

Measured on 2026-07-03 (`sam enrich --evaluate`, FinBERT):

| Metric | Value | Gate |
|---|---|---|
| Macro-F1 | **0.739** | ≥ 0.70 ✅ |
| Accuracy | 0.762 | — |
| F1 neutral / negative / positive | 0.84 / 0.76 / 0.62 | — |

The gate is a permanent pytest
(`tests/test_nlp_evaluate.py::test_finbert_meets_macro_f1_gate`), skipped
where the `nlp` extra is absent (CI) and enforced locally, so sentiment
quality cannot silently regress. Metric math itself is model-free and always
tested.

**Known failure pattern (documented, not hidden):** FinBERT leans
"up-is-good" — macro headlines where a *rise* is bad news ("Core inflation
hit 3.4%, highest since 2023", "Wholesale prices rose more than expected")
score *positive*. 6 of the 15 misses follow this pattern. Mitigations belong
in P5 (e.g. sign-flipping macro-inflation contexts or entity-scoped
weighting), not in silent label edits.

## Topics

`sam topics` fits BERTopic over every document embedded by the configured
model, using the **stored** vectors (no re-encoding). Each run writes a fresh
`topic_model_version` (append-only — past SAI values were computed against
past topic versions; point-in-time rule). Outlier documents (BERTopic's `-1`
bucket) are deliberately left unassigned.

Live fit on the current corpus (260 docs, 2026-07-03): **2 topics,
31 outliers, 229 assignments** — `0_ai_stocks_traders_said` (market news,
151 docs) and `1_hn_code_claude_email` (Hacker News tech, 78 docs). This is
honest small-corpus behavior: the architecture's "topics stable" ambition
needs corpus growth (weeks of scheduled ingestion), which is why
`nlp.topics_min_docs` (default 50) skips the fit entirely below a floor
rather than fitting degenerate clusters. Expect topic granularity to improve
naturally as documents accumulate; revisit after ~1k documents.

## Data quality

`sam dq` gains `enrichment_coverage`: share of documents enriched, with the
unenriched backlog in `details`; warns only when a sizable corpus has zero
sentiment rows (pipeline rot), mirroring `resolution_coverage`.

## Operations

```bash
SAM_DB__PORT=5433 uv run sam enrich             # incremental (cron-safe)
SAM_DB__PORT=5433 uv run sam enrich --all       # re-enrich after model change
uv run sam enrich --evaluate                    # macro-F1 gate (no DB needed)
SAM_DB__PORT=5433 uv run sam topics             # versioned topic fit (weekly)
SAM_DB__PORT=5433 uv run sam dq                 # quality checks incl. coverage
```

First `sam enrich` downloads models from Hugging Face (~500 MB, cached under
`~/.cache/huggingface`). The `nlp` extra is required at runtime
(`uv sync --extra nlp`); without it the CLI exits with a clear hint instead
of a traceback.

## Known limitations (deliberate, documented)

- **FinBERT's up-is-good bias** on macro-inflation headlines (measured above).
- **Document-level sentiment only** — one label per document, weighted per
  entity by link confidence downstream; no aspect-based scoring yet.
- **Topic instability at current corpus size** — 2 coarse topics on 260 docs;
  versioned runs make future improvement measurable rather than silent.
- **Topic runs are corpus-global** (not incremental); acceptable at current
  scale, revisit (online topic updates / MinHash-style optimizations) with
  Reddit-scale volume.
- **English-only** models; `documents.lang` exists for future filtering.
