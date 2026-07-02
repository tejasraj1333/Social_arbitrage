"""Loader for config/sources.yaml (shared by all recon collectors)."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import yaml

from sam.core.config import CONFIG_DIR

SOURCES_YAML = CONFIG_DIR / "sources.yaml"


@lru_cache
def load_sources() -> dict[str, Any]:
    """Parse config/sources.yaml once and cache it."""
    with SOURCES_YAML.open(encoding="utf-8") as fh:
        data: dict[str, Any] = yaml.safe_load(fh) or {}
    return data
