"""Canonical content hashing — the dedup anchor for idempotent ingestion.

The hash must be *stable across fetches* of the same content (so re-running a
collector is a no-op) and *distinct across different content*. Canonicalization
rules: None -> empty, NFC unicode normalization, whitespace collapsed, parts
joined with an ASCII unit separator so part boundaries can't be forged by
concatenation ("ab","c" vs "a","bc").

Engagement metrics (score, comments) must NEVER be part of the hash: they
change on every fetch and would defeat dedup.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

_WS_RE = re.compile(r"\s+")
_SEP = "\x1f"  # ASCII unit separator: cannot appear in collapsed text


def _canonicalize(part: str | None) -> str:
    if part is None:
        return ""
    return _WS_RE.sub(" ", unicodedata.normalize("NFC", part)).strip()


def content_hash(*parts: str | None) -> str:
    """SHA-256 hex digest over the canonicalized parts (order-sensitive)."""
    joined = _SEP.join(_canonicalize(p) for p in parts)
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()
