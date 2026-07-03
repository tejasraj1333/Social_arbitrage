"""Bronze (raw) lake writer — immutable, source-shaped, append-only.

Layout (medallion, per the Phase-1 blueprint):

    data/00_raw/<source>/dt=YYYY-MM-DD/<source>-<utc-ts>-<nonce>.jsonl.gz
    data/00_raw/_manifests/<same-stem>.json

Every write produces a fresh uniquely-named file (never overwrites) plus a
manifest recording row count, payload sha256 and timestamps — the provenance
needed to re-derive downstream layers from raw. `dt` is the *ingestion* date
(UTC): bronze partitions by when we learned the data, not when it happened
(point-in-time rule).
"""

from __future__ import annotations

import gzip
import hashlib
import json
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sam.core.config import PROJECT_ROOT
from sam.core.logging import get_logger

log = get_logger("ingestion.lake")

RAW_LAKE_DIR = PROJECT_ROOT / "data" / "00_raw"  # module global; tests monkeypatch


@dataclass(slots=True)
class LakeArtifact:
    """Where a raw batch landed, and its provenance manifest."""

    path: str
    manifest_path: str
    rows: int
    sha256: str


def write_raw(source_name: str, records: list[dict[str, Any]]) -> LakeArtifact | None:
    """Persist one fetch's raw records; returns None for an empty batch."""
    if not records:
        return None

    now = datetime.now(tz=UTC)
    stem = f"{source_name}-{now.strftime('%Y%m%dT%H%M%SZ')}-{uuid.uuid4().hex[:8]}"
    partition = RAW_LAKE_DIR / source_name / f"dt={now.date().isoformat()}"
    manifests = RAW_LAKE_DIR / "_manifests"
    partition.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)

    payload = "\n".join(json.dumps(r, ensure_ascii=False, default=str) for r in records) + "\n"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()

    data_path = partition / f"{stem}.jsonl.gz"
    with gzip.open(data_path, "wt", encoding="utf-8") as fh:
        fh.write(payload)

    manifest_path = manifests / f"{stem}.json"
    manifest = {
        "source": source_name,
        "path": str(data_path),
        "rows": len(records),
        "sha256": digest,
        "written_at": now.isoformat(),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    log.info("lake_written", source=source_name, rows=len(records), path=str(data_path))
    return LakeArtifact(
        path=str(data_path), manifest_path=str(manifest_path), rows=len(records), sha256=digest
    )


def read_raw(path: str | Path) -> list[dict[str, Any]]:
    """Round-trip helper (reprocessing/tests): load one bronze file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]
