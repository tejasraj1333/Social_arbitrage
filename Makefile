.PHONY: install lint format format-check type test check migrate revision up down serve

install:        ## Sync deps (incl. dev) via uv
	uv sync --extra dev

lint:           ## Ruff lint
	uv run ruff check src tests

format:         ## Ruff format
	uv run ruff format src tests

format-check:   ## Ruff format check (no writes)
	uv run ruff format --check src tests

type:           ## mypy strict
	uv run mypy src

test:           ## Run tests with coverage
	uv run pytest

check: lint format-check type test  ## Full local gate (mirrors CI)

migrate:        ## Apply migrations
	uv run alembic upgrade head

revision:       ## Autogenerate a migration: make revision m="message"
	uv run alembic revision --autogenerate -m "$(m)"

serve:          ## Run API locally with reload
	uv run uvicorn sam.api.app:app --reload

up:             ## Start docker stack
	docker compose up --build

down:           ## Stop docker stack
	docker compose down
