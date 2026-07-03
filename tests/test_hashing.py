"""Unit tests for canonical content hashing."""

from __future__ import annotations

from sam.ingestion.hashing import content_hash


def test_deterministic() -> None:
    assert content_hash("rss", "https://x/a", "Title") == content_hash(
        "rss", "https://x/a", "Title"
    )


def test_distinct_content_distinct_hash() -> None:
    assert content_hash("rss", "https://x/a") != content_hash("rss", "https://x/b")
    assert content_hash("rss", "u", "Title A") != content_hash("rss", "u", "Title B")


def test_part_boundaries_cannot_be_forged() -> None:
    assert content_hash("ab", "c") != content_hash("a", "bc")
    assert content_hash("ab") != content_hash("ab", "")


def test_whitespace_and_unicode_canonicalized() -> None:
    assert content_hash("Tesla   surges\n today ") == content_hash("Tesla surges today")
    # NFC (precomposed e-acute) vs NFD (e + combining acute) must hash identically.
    assert content_hash("café") == content_hash("café")


def test_none_equals_empty() -> None:
    assert content_hash("rss", None, "t") == content_hash("rss", "", "t")


def test_output_shape() -> None:
    digest = content_hash("anything")
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)
