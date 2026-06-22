# Multi-stage build. Uses uv for fast, reproducible installs from uv.lock.
FROM python:3.11-slim AS base
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

# uv from its official distroless image (pinned).
COPY --from=ghcr.io/astral-sh/uv:0.5.11 /uv /usr/local/bin/uv

WORKDIR /app

# Install deps first (cache layer) — only core runtime deps for the API image.
COPY pyproject.toml uv.lock* ./
RUN uv sync --no-install-project --no-dev

# App source.
COPY src ./src
COPY config ./config
COPY migrations ./migrations
COPY alembic.ini ./
RUN uv sync --no-dev

ENV PATH="/app/.venv/bin:$PATH"

EXPOSE 8000
CMD ["uvicorn", "sam.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
