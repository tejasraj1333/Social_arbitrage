"""Phase-1 recon collector contract.

Distinct from the production `sam.ingestion.base.Collector` (fetch/normalize/
persist -> Postgres). Recon collectors answer one question per source: *can it
supply usable data?* The shape is therefore:

    fetch()        -> list of raw records (network; may raise CredentialsMissing)
    validate()     -> ValidationReport (record count + schema completeness)
    save_sample()  -> write raw + a capped sample to data/raw|sample/<source>.*

`run()` ties them together and returns a `ReconResult` (the row the scorecard
consumes). Subclasses implement `fetch`; `validate`/`save_sample`/`run` have
sensible defaults a subclass can override (Yahoo overrides save_sample -> CSV).
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from typing import Any, Literal

from sam.core.config import PROJECT_ROOT
from sam.core.errors import CredentialsMissing
from sam.core.logging import get_logger

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SAMPLE_DIR = DATA_DIR / "sample"

# ok = usable data fetched; empty = ran but nothing came back;
# needs_credentials = auth absent (built, not yet proven); error = unexpected failure.
ReconStatus = Literal["ok", "empty", "needs_credentials", "error"]

_NULLISH: tuple[Any, ...] = (None, "", [], {})


@dataclass(slots=True)
class ValidationReport:
    """Schema/quality summary for one fetch."""

    record_count: int
    required_fields: list[str]
    field_completeness: dict[str, float]  # per-field fraction of non-null values
    schema_completeness: float  # mean completeness across required fields (0..1)
    issues: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.record_count > 0 and not self.issues


@dataclass(slots=True)
class ReconResult:
    """One source's recon outcome — the scorecard's input row."""

    source_name: str
    status: ReconStatus
    record_count: int
    schema_completeness: float
    freshness_hours: float | None
    sample_path: str | None
    raw_path: str | None
    estimated_monthly_volume: int | None = None
    detail: str = ""
    validation: ValidationReport | None = None

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # validation nests as a dict via asdict; keep it JSON-friendly.
        return d


class ReconCollector(ABC):
    """Abstract base for all Phase-1 recon collectors."""

    #: short, filesystem-safe id (also the sample/raw filename stem).
    source_name: str
    #: fields every record must carry for the source to be "usable".
    required_fields: list[str]
    #: record key holding the publish time (epoch seconds, datetime, or ISO str);
    #: used to compute freshness. None -> freshness not applicable (e.g. price bars).
    timestamp_field: str | None = None
    #: how many records to persist into the committed sample file.
    sample_size: int = 100

    def __init__(self) -> None:
        self.log = get_logger(f"recon.{self.source_name}")

    # --- contract -------------------------------------------------------------

    @abstractmethod
    def fetch(self) -> list[dict[str, Any]]:
        """Pull raw records from the source. Raise CredentialsMissing if unauthenticated."""

    def validate(self, records: Sequence[dict[str, Any]]) -> ValidationReport:
        """Compute record count + per-field completeness over `required_fields`.

        A field present-but-empty counts as missing. A required field absent from
        *every* record is a hard schema issue; partial nulls are reported, not fatal.
        """
        n = len(records)
        completeness: dict[str, float] = {}
        for f in self.required_fields:
            if n == 0:
                completeness[f] = 0.0
                continue
            non_null = sum(1 for r in records if r.get(f) not in _NULLISH)
            completeness[f] = non_null / n

        schema = fmean(completeness.values()) if completeness else 0.0
        issues: list[str] = []
        if n == 0:
            issues.append("no records fetched")
        for f, cov in completeness.items():
            if cov == 0.0 and n > 0:
                issues.append(f"required field '{f}' absent from all records")
        return ValidationReport(
            record_count=n,
            required_fields=list(self.required_fields),
            field_completeness=completeness,
            schema_completeness=schema,
            issues=issues,
        )

    def save_sample(self, records: Sequence[dict[str, Any]]) -> tuple[str | None, str | None]:
        """Write all records to data/raw and a capped sample to data/sample (JSONL).

        Returns (raw_path, sample_path) as strings, or (None, None) if empty.
        """
        if not records:
            return None, None
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        raw_path = RAW_DIR / f"{self.source_name}.jsonl"
        sample_path = SAMPLE_DIR / f"{self.source_name}.jsonl"
        self._write_jsonl(raw_path, records)
        self._write_jsonl(sample_path, records[: self.sample_size])
        return str(raw_path), str(sample_path)

    # --- orchestration --------------------------------------------------------

    def run(self) -> ReconResult:
        self.log.info("recon_start", source=self.source_name)
        try:
            records = list(self.fetch())
        except CredentialsMissing as exc:
            self.log.warning("recon_needs_credentials", source=self.source_name, detail=str(exc))
            return ReconResult(
                source_name=self.source_name,
                status="needs_credentials",
                record_count=0,
                schema_completeness=0.0,
                freshness_hours=None,
                sample_path=None,
                raw_path=None,
                detail=str(exc),
            )
        except Exception as exc:
            # One broken source must not sink the whole recon run: surface it as
            # a scorecard row with status="error" instead of crashing the process.
            self.log.error("recon_error", source=self.source_name, error=str(exc), exc_info=True)
            return ReconResult(
                source_name=self.source_name,
                status="error",
                record_count=0,
                schema_completeness=0.0,
                freshness_hours=None,
                sample_path=None,
                raw_path=None,
                detail=f"{type(exc).__name__}: {exc}",
            )

        report = self.validate(records)
        raw_path, sample_path = self.save_sample(records)
        freshness = self.freshness_hours(records)
        monthly_volume = self.estimated_monthly_volume(records)
        status: ReconStatus = "ok" if records else "empty"
        result = ReconResult(
            source_name=self.source_name,
            status=status,
            record_count=report.record_count,
            schema_completeness=round(report.schema_completeness, 4),
            freshness_hours=freshness,
            sample_path=sample_path,
            raw_path=raw_path,
            estimated_monthly_volume=monthly_volume,
            detail="; ".join(report.issues),
            validation=report,
        )
        self.log.info(
            "recon_done",
            source=self.source_name,
            status=status,
            records=report.record_count,
            schema_completeness=result.schema_completeness,
            freshness_hours=freshness,
            sample=sample_path,
        )
        return result

    # --- helpers --------------------------------------------------------------

    def freshness_hours(self, records: Sequence[dict[str, Any]]) -> float | None:
        """Hours since the most recent record, via `timestamp_field`. None if N/A."""
        if not records or self.timestamp_field is None:
            return None
        epochs = [
            e for r in records if (e := self._to_epoch(r.get(self.timestamp_field))) is not None
        ]
        if not epochs:
            return None
        now = datetime.now(tz=UTC).timestamp()
        return round((now - max(epochs)) / 3600.0, 2)

    def estimated_monthly_volume(self, records: Sequence[dict[str, Any]]) -> int | None:
        """Extrapolate records/month from the interquartile record density.

        Uses the span between the 25th and 75th timestamp percentiles (which hold
        ~half the records) rather than min..max — robust to stale/evergreen feed
        entries that would otherwise blow up the span. Order-of-magnitude only;
        None when not time-stamped or too few records.
        """
        if not records or self.timestamp_field is None:
            return None
        epochs = sorted(
            e for r in records if (e := self._to_epoch(r.get(self.timestamp_field))) is not None
        )
        n = len(epochs)
        if n < 4:
            return None
        q1 = epochs[int(0.25 * (n - 1))]
        q3 = epochs[int(0.75 * (n - 1))]
        span_hours = (q3 - q1) / 3600.0
        if span_hours <= 0:
            return None
        rate_per_hour = (0.5 * n) / span_hours  # ~half the records lie in [q1, q3]
        return int(round(rate_per_hour * 24 * 30))

    @staticmethod
    def _to_epoch(value: Any) -> float | None:
        if value is None:
            return None
        if isinstance(value, bool):  # guard: bool is an int subclass
            return None
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                value = datetime.fromisoformat(value.replace("Z", "+00:00"))
            except ValueError:
                return None
        if isinstance(value, datetime):
            # Source timestamps are UTC by convention; a naive value must not be
            # reinterpreted in the machine's local timezone (point-in-time rule).
            dt: datetime = value if value.tzinfo is not None else value.replace(tzinfo=UTC)
            return dt.timestamp()
        return None

    @staticmethod
    def _write_jsonl(path: Path, records: Sequence[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for rec in records:
                fh.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
