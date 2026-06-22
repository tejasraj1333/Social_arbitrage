# Social Arbitrage Model (SAM)

> Detect attention & sentiment shifts before they surface in earnings or price.

Markets often react to trends after they become obvious. SAM measures the
**rate of change** of public attention and sentiment per entity and converts it
into a tradable signal — the **Social Arbitrage Index (SAI)** — to capture the
window between *"the crowd notices"* and *"the market prices it in."*

## ⚠️ Disclaimer

This is a **research and educational** project. It is **not investment advice**
and performs **no live trading**. Respect each data source's Terms of Service.

## Key concept: the Social Arbitrage Index (SAI)

A daily, per-entity composite of four sub-signals:

| Sub-signal | Captures |
|---|---|
| Mention growth | Is this entity being talked about more? |
| Sentiment momentum | Is the *tone* improving or deteriorating? |
| Topic velocity | How fast is an emerging narrative spreading? |
| Engagement growth | Are people reacting more strongly (votes, comments)? |

Each is normalized against a trailing window and weighted into `sai_score`.
The signal is **point-in-time correct** (every record stores when it was *known*,
not just when it happened) so backtests are free of look-ahead bias.

## Architecture at a glance

```
ingest ─► process/entity-resolution ─► nlp (sentiment, embeddings, topics)
        ─► SAI ─► backtest/validation ─► forecasting ─► reporting ─► API
```

Deterministic core (reproducible signals) with an optional future LangGraph
agent layer that *narrates and routes* but never computes the signal.
Full design: [`docs/architecture.md`](docs/architecture.md).

## Tech stack

Python 3.11 · uv · FastAPI · PostgreSQL + **pgvector** · SQLAlchemy/Alembic ·
structlog · FinBERT + sentence-transformers + BERTopic · XGBoost/LightGBM
(→ LSTM/TFT) · **hybrid LLM** (local models for bulk, Claude for report narration)
· Docker.

## Quickstart

Prerequisites: Python 3.11+, [uv](https://docs.astral.sh/uv/), Docker.

```bash
# 1. Install uv (if needed)
#    Windows (PowerShell): irm https://astral.sh/uv/install.ps1 | iex
#    macOS/Linux:          curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install deps
uv sync --extra dev

# 3. Configure
cp .env.example .env        # then fill in secrets

# 4. Run the full stack (API + Postgres/pgvector + migrations)
docker compose up --build
# API docs: http://localhost:8000/docs   health: http://localhost:8000/health

# --- or develop locally ---
uv run alembic upgrade head     # needs a running Postgres
uv run uvicorn sam.api.app:app --reload
```

## Configuration

Three layers, lowest precedence first: `config/base.yaml` (committed,
non-secret) → environment variables (`SAM_*`, nested via `__`) → `.env` (local).
See [`.env.example`](.env.example) and `src/sam/core/config.py`.

## Development

```bash
make install   # uv sync --extra dev
make check     # ruff + mypy + pytest (mirrors CI)
make serve     # local API with reload
make test      # tests + coverage
```

Pre-commit: `uv run pre-commit install`.

## Project layout

```
src/sam/        core/ ingestion/ processing/ nlp/ signals/
                forecasting/ reporting/ backtest/ storage/ api/ pipelines/
config/         base.yaml · sources.yaml · logging.yaml
migrations/     Alembic
tests/          unit + integration
docs/           architecture.md, sai_methodology.md, adr/
```

## Roadmap

M0 Skeleton ✅ · M1 Full ingestion · M2 Entity resolution · M3 NLP ·
M4 SAI · **M5 Validation (kill-gate)** · M6 Forecasting · M7 Reporting ·
M8 Serving/hardening · M9 Agentic (future). See `docs/architecture.md`.

## License

MIT.
