"""Configuration management.

Layering, highest precedence first:
  1. environment variables     secrets & per-env overrides (SAM_ prefix)
  2. .env file (local only)     convenience for local dev
  3. config/base.yaml          non-secret defaults, committed

Nested settings use a "__" delimiter, e.g. SAM_DB__HOST -> settings.db.host.
Env always wins over the committed YAML so a deployment can override any
default without editing tracked files. Access the singleton via
`get_settings()` (cached).
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CONFIG_DIR = PROJECT_ROOT / "config"
BASE_YAML = CONFIG_DIR / "base.yaml"


class DatabaseSettings(BaseModel):
    host: str = "localhost"
    port: int = 5432
    user: str = "sam"
    password: str = "sam"
    name: str = "sam"

    @property
    def url(self) -> str:
        """SQLAlchemy URL (psycopg3 driver)."""
        return (
            f"postgresql+psycopg://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"
        )


class RedditSettings(BaseModel):
    client_id: str = ""
    client_secret: str = ""
    user_agent: str = "social-arbitrage-model/0.1"


class KaggleSettings(BaseModel):
    username: str = ""
    key: str = ""


class LLMSettings(BaseModel):
    anthropic_api_key: str = ""
    report_model: str = "claude-opus-4-8"


class NLPSettings(BaseModel):
    """NLP enrichment models (Phase 4). Model ids are recorded on every row
    they produce, so changing one here starts writing under the new id rather
    than silently mixing outputs. The embedding model's width must match the
    schema (see sam.storage.models.EMBEDDING_DIM)."""

    sentiment_model: str = "ProsusAI/finbert"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    # "cpu" (default) or "cuda". Both configured models fit small GPUs.
    device: str = "cpu"
    batch_size: int = 32
    # Below this many enriched documents a topic fit is skipped (UMAP/HDBSCAN
    # degenerate on tiny corpora); mirrors DQ's "insufficient history" honesty.
    topics_min_docs: int = 50


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="SAM_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        yaml_file=BASE_YAML,
        yaml_file_encoding="utf-8",
        extra="ignore",
    )

    env: str = "local"
    log_level: str = "INFO"
    log_json: bool = True

    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    reddit: RedditSettings = Field(default_factory=RedditSettings)
    kaggle: KaggleSettings = Field(default_factory=KaggleSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    nlp: NLPSettings = Field(default_factory=NLPSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Priority high -> low. YAML is the lowest layer so env/.env override it.
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(settings_cls),
            file_secret_settings,
        )


@lru_cache
def get_settings() -> Settings:
    """Return the cached Settings singleton (env > .env > YAML defaults)."""
    return Settings()
