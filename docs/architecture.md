# SAM — Architecture & Design

This document is the durable design reference. The conversational blueprint that
produced it covered ten areas (roadmap, README, structure, milestones, schema,
agents, API, training, evaluation, deployment); this file keeps the decisions
that future contributors need.

## 1. Thesis

Public attention shifts lead fundamentals. By measuring the *rate of change* of
attention and sentiment per entity and converting it into a signal (SAI), we aim
to capture the window between crowd awareness and market pricing.

The hard parts (and where effort is concentrated): **entity resolution**,
**avoiding look-ahead bias**, and **proving the signal is real, not overfit**.

## 2. Roadmap & milestones

| Phase | Outcome | Gate |
|---|---|---|
| M0 Foundations | repo, uv, Docker, config, logging, CI, migrations | `docker compose up` green |
| M1 Ingestion (full) | Reddit + RSS + Yahoo + Kaggle → raw store | 7 days flowing, idempotent |
| M2 Entity resolution | text → ticker mapping | ≥90% precision on labeled sample |
| M3 NLP | FinBERT sentiment + embeddings + BERTopic | sentiment validated, topics stable |
| M4 SAI | 4 sub-signals → composite | deterministic rebuild from raw |
| **M5 Validation** | walk-forward, IC, decay | **kill-gate: significant forward IC** |
| M6 Forecasting | XGBoost/LightGBM → LSTM/TFT | beats naive baseline OOS |
| M7 Reporting | LLM narrative over signals | zero hallucinated numbers |
| M8 Serving | FastAPI, vector search, monitoring | p95 latency target, authn |
| M9 Agentic (future) | LangGraph orchestration | gated on M1–M8 |

M5 is a hard kill-gate: if the signal has no forward predictive power, later
phases are reprioritized rather than built on noise.

## 3. Target database schema

Postgres holds structured metadata + time series; **pgvector** holds embeddings
in the same database (chosen vector store — one fewer service to operate).

Critical rule: every document carries both `published_at` (event time) and
`ingested_at` (known time). Training and backtests join on *known* time only.

```
entities(id, ticker, name, sector, aliases[], active, created_at)
sources(id, type, name, config_ref)
documents(id, source_id, external_id, url, author,
          content_hash UNIQUE, raw_text, lang,
          published_at, ingested_at, engagement JSONB)
document_entities(document_id, entity_id, confidence, method, resolved_at)
sentiment_scores(document_id, model, label, score, scored_at)
topics(id, topic_model_version, label, keywords[], created_at)
document_topics(document_id, topic_id, probability)
embeddings(document_id, model, vector VECTOR(384), embedded_at)  -- pgvector
sai_daily(entity_id, date, mention_growth, sentiment_momentum,
          topic_velocity, engagement_growth, sai_score, sai_rank, computed_at)
market_data(entity_id, date, open, high, low, close, adj_close, volume)
forecasts(entity_id, date, horizon, model_version, yhat, yhat_low, yhat_high)
reports(id, entity_id, period, content, model, created_at)
ingestion_runs(id, source_id, started_at, finished_at, rows, status, error)
data_quality_checks(id, check_name, source_name, status, value, threshold,
                    details JSONB, ran_at)
model_registry(id, name, version, metrics JSONB, path, created_at)
```

M0 ships only `entities` + the `vector` extension; the rest land per milestone
via `alembic revision --autogenerate`.

## 4. Agent architecture (future)

Keep the core pipeline deterministic. Expose each service as a **tool**; a future
LangGraph supervisor routes between a Research agent, Sentiment/Topic analyst,
Forecast agent, and Report writer. **Guardrail: agents narrate and route; they
never compute the signal.** All numbers come from deterministic services.

## 5. API architecture

FastAPI, versioned under `/v1`. Split read API (fast, cached) from command API
(async pipeline triggers via a worker queue).

```
GET  /health  /ready
GET  /v1/entities
GET  /v1/signals/{ticker}?from&to        SAI series + components
GET  /v1/signals/top?date&n              ranked movers
GET  /v1/forecasts/{ticker}?horizon
POST /v1/search                          semantic search (pgvector)
GET  /v1/reports/{ticker}/latest
POST /v1/reports/{ticker}                async -> job id
POST /v1/pipelines/run                   async, admin
GET  /v1/jobs/{id}
```

Cross-cutting: API-key/JWT auth + roles, rate limiting, request-id logging,
Pydantic schemas separate from ORM models, pagination, heavy inference offloaded
to workers.

## 6. Model training pipeline

1. Feature build — point-in-time join of SAI components + lags/rolling stats +
   market features, all as-known-at-date.
2. Target build — forward return over horizon `h`.
3. Split — **walk-forward / expanding window** (never random shuffle on time series).
4. Train — GBMs first (XGBoost/LightGBM, interpretable, strong on tabular);
   LSTM/TFT only if they beat GBMs out-of-sample (TFT adds quantile/uncertainty
   outputs and temporal attention).
5. Validate → 6. Register (model_registry) → 7. Promote only if it beats incumbent
   on a reserved holdout.

Dominant risk: **leakage**. Enforce point-in-time features + temporal CV.

## 7. Evaluation metrics

Component: sentiment F1 vs labeled set; topic coherence (c_v)/stability;
forecast OOS RMSE/MAE + **directional accuracy**.

Signal (the real test): **Information Coefficient** (rank-corr of SAI vs forward
returns), IC decay by horizon, decile spread, backtest Sharpe / max drawdown /
hit rate / turnover **net of transaction costs**, robustness across regimes.

Guard against p-hacking: reserve a true holdout period never touched during
development; always include transaction costs.

## 8. Deployment

Split scheduled **batch** (ingest → build SAI → retrain) from online **serving**
(API), scaled and failed independently. One image, multiple entrypoints
(api / worker / scheduler). Managed Postgres+pgvector, object storage for model
artifacts, Redis + worker queue for async jobs.

Observability: structured JSON logs, Prometheus + Grafana (ingestion freshness,
pipeline success, API latency, **model IC over time**), alerting on **data
staleness** (not just errors), Sentry. CI/CD via GitHub Actions; migrations as a
gated deploy step; secrets injected at runtime, never baked into images.

## 9. Cross-cutting risks

1. Look-ahead bias — designed against in schema (`ingested_at`) and temporal CV.
2. Entity-resolution quality — its own milestone (M2); garbage in → garbage SAI.
3. Source ToS/licensing — Reddit API, RSS, Kaggle, yfinance (unofficial, may break;
   isolated behind the `Collector` interface).
4. The signal might not exist — M5 kill-gate exists to find out early.
5. Reproducibility vs agents — deterministic code computes, agents only narrate.
