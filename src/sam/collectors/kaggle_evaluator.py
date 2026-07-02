"""Kaggle dataset evaluator (needs credentials, metadata-only).

Phase-1 recon does NOT download datasets — it only queries metadata (ref, title,
size, license, last-updated, popularity) for candidates matching the configured
search terms, so we can judge which Kaggle datasets are worth ingesting later.

The `kaggle` package authenticates eagerly *at import* and calls sys.exit() when
credentials are absent, so the import is sandboxed: its banner is suppressed and
the SystemExit is converted to CredentialsMissing -> run() reports
status="needs_credentials" instead of killing the process.

Record schema: ref, title, size, license, last_updated (+ popularity, search_term).
"""

from __future__ import annotations

import contextlib
import io
import os
from typing import Any

from sam.core.config import get_settings
from sam.core.errors import CredentialsMissing
from sam.recon.collector_base import ReconCollector
from sam.recon.sources import load_sources


class KaggleEvaluator(ReconCollector):
    source_name = "kaggle"
    required_fields = ["ref", "title", "size", "license", "last_updated"]
    timestamp_field = None  # metadata spans many datasets; per-dataset freshness N/A
    sample_size = 100

    def __init__(
        self,
        search_terms: list[str] | None = None,
        per_term: int = 20,
    ) -> None:
        super().__init__()
        cfg = load_sources().get("kaggle", {})
        self.search_terms = (
            search_terms if search_terms is not None else cfg.get("search_terms", [])
        )
        self.per_term = per_term

    def fetch(self) -> list[dict[str, Any]]:
        api = self._api()  # raises CredentialsMissing when unauthenticated
        records: list[dict[str, Any]] = []
        seen: set[str] = set()
        for term in self.search_terms:
            try:
                results = api.dataset_list(search=term, sort_by="hottest")
            except Exception as exc:  # one bad term shouldn't sink the rest
                self.log.warning("kaggle_search_error", term=term, error=str(exc))
                continue
            kept = 0
            for ds in results:
                ref = str(self._attr(ds, ("ref",)) or ds)
                if ref in seen:
                    continue
                seen.add(ref)
                records.append(self._to_record(ds, term))
                kept += 1
                if kept >= self.per_term:
                    break
            self.log.info("kaggle_term_done", term=term, kept=kept)
        return records

    def _api(self) -> Any:
        """Import + authenticate Kaggle, sandboxing its eager-auth side effects."""
        self._export_credentials()
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                import kaggle
        except SystemExit as exc:  # kaggle calls sys.exit() when creds absent
            raise CredentialsMissing(
                "Kaggle credentials missing - place kaggle.json in ~/.kaggle/ "
                "(or set KAGGLE_USERNAME / KAGGLE_KEY). "
                "See https://www.kaggle.com/settings/api"
            ) from exc
        except Exception as exc:  # any other import/auth failure
            raise CredentialsMissing(f"Kaggle API unavailable: {exc}") from exc
        return kaggle.api

    @staticmethod
    def _export_credentials() -> None:
        """Bridge SAM_KAGGLE__* settings to the env vars the kaggle client reads.

        The kaggle package only understands ~/.kaggle/kaggle.json or bare
        KAGGLE_USERNAME/KAGGLE_KEY env vars; without this bridge the documented
        SAM settings would be silently ignored. setdefault keeps any explicit
        env vars / kaggle.json the user already has as the source of truth.
        """
        creds = get_settings().kaggle
        if creds.username and creds.key:
            os.environ.setdefault("KAGGLE_USERNAME", creds.username)
            os.environ.setdefault("KAGGLE_KEY", creds.key)

    @staticmethod
    def _attr(obj: Any, names: tuple[str, ...], default: Any = None) -> Any:
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                return value
        return default

    @classmethod
    def _to_record(cls, ds: Any, term: str) -> dict[str, Any]:
        return {
            "ref": str(cls._attr(ds, ("ref",), ds)),
            "title": cls._attr(ds, ("title",)),
            "size": cls._attr(ds, ("size", "totalBytes")),
            "license": cls._attr(ds, ("licenseName", "license_name")),
            "last_updated": str(cls._attr(ds, ("lastUpdated", "last_updated")) or ""),
            "download_count": cls._attr(ds, ("downloadCount", "download_count")),
            "vote_count": cls._attr(ds, ("voteCount", "vote_count")),
            "usability_rating": cls._attr(ds, ("usabilityRating", "usability_rating")),
            "search_term": term,
        }
