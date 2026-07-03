"""Deterministic text→ticker entity resolver (Phase 3 / architecture M2).

Pure functions over an in-memory entity dictionary — no DB access here, so
the matching rules are exhaustively unit-testable and the resolver stays
reproducible (a design invariant: signals must rebuild deterministically).

Rule tiers, strongest wins per entity:

  cashtag   ``$NVDA``                       confidence 1.0
  ticker    bare ``NVDA`` token              confidence 0.9
  alias     ``Nvidia`` / full company name   confidence 0.8

Bare-ticker matching is case-sensitive (lowercase "nvda" in prose is more
likely noise) and skipped entirely for tickers that are common English words
(``AMBIGUOUS_TICKERS``) — those still match via cashtag or alias. Alias and
name matching is case-insensitive on word boundaries. Confidence is a
*weight* for downstream signals, not a filter (quarantine-don't-delete
philosophy applies to links too).
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

CASHTAG_CONFIDENCE = 1.0
TICKER_CONFIDENCE = 0.9
ALIAS_CONFIDENCE = 0.8

# Tickers that collide with everyday English words: a bare-token match would
# produce false positives at scale, so they resolve only via cashtag or alias.
# Curated; extend as the universe grows (e.g. Agilent "A", Realty Income "O").
AMBIGUOUS_TICKERS: frozenset[str] = frozenset(
    {
        "A",
        "ALL",
        "AN",
        "ANY",
        "ARE",
        "AT",
        "BE",
        "BIG",
        "BY",
        "CAN",
        "CAR",
        "DD",
        "DO",
        "EAT",
        "FOR",
        "FUN",
        "GO",
        "GOOD",
        "HAS",
        "HE",
        "IT",
        "KEY",
        "LOVE",
        "LOW",
        "MAN",
        "NEW",
        "NEXT",
        "NICE",
        "NOW",
        "ON",
        "ONE",
        "OPEN",
        "OR",
        "OUT",
        "PLAY",
        "REAL",
        "RUN",
        "SEE",
        "SO",
        "STAY",
        "TECH",
        "TV",
        "UP",
        "WELL",
        "YOU",
    }
)


@dataclass(frozen=True, slots=True)
class EntityRef:
    """The slice of an entities row the matcher needs (built by the pipeline)."""

    entity_id: int
    ticker: str
    name: str
    aliases: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class EntityMatch:
    """One resolved mention: which entity, how confidently, via which rule."""

    entity_id: int
    ticker: str
    confidence: float
    method: str  # "cashtag" | "ticker" | "alias"


def _phrase_pattern(phrase: str, *, ignore_case: bool) -> re.Pattern[str]:
    """Compile a word-bounded pattern for a literal phrase.

    Lookarounds instead of ``\\b``: names like "Amazon.com, Inc." end in a
    non-word char, where ``\\b`` never matches (no \\w on either side) — the
    lookarounds only assert no *word* character is adjacent.
    """
    return re.compile(
        rf"(?<!\w){re.escape(phrase)}(?!\w)",
        re.IGNORECASE if ignore_case else 0,
    )


@dataclass(frozen=True, slots=True)
class _CompiledEntity:
    ref: EntityRef
    cashtag: re.Pattern[str]
    ticker: re.Pattern[str] | None  # None when the ticker is ambiguous
    aliases: tuple[re.Pattern[str], ...]


class EntityMatcher:
    """Compiled matcher over a fixed entity dictionary.

    Build once per resolution run (compilation is O(universe)), then call
    :meth:`match` per document.
    """

    def __init__(self, entities: Sequence[EntityRef]) -> None:
        self._compiled: list[_CompiledEntity] = []
        for ref in entities:
            alias_phrases = {ref.name, *ref.aliases}
            self._compiled.append(
                _CompiledEntity(
                    ref=ref,
                    # $ sigil disambiguates, so cashtags match case-insensitively.
                    cashtag=_phrase_pattern(f"${ref.ticker}", ignore_case=True),
                    ticker=(
                        None
                        if ref.ticker.upper() in AMBIGUOUS_TICKERS
                        else _phrase_pattern(ref.ticker, ignore_case=False)
                    ),
                    aliases=tuple(
                        _phrase_pattern(phrase, ignore_case=True)
                        for phrase in sorted(alias_phrases)
                        if phrase
                    ),
                )
            )

    def match(self, *texts: str | None) -> list[EntityMatch]:
        """Resolve entity mentions across the given text fields (title, body, ...).

        Per entity the strongest matching rule wins; results are sorted by
        confidence (desc) then ticker for deterministic output.
        """
        corpus = " \n ".join(unicodedata.normalize("NFC", t) for t in texts if t)
        if not corpus:
            return []

        matches: list[EntityMatch] = []
        for comp in self._compiled:
            method = self._strongest_method(comp, corpus)
            if method is None:
                continue
            confidence = {
                "cashtag": CASHTAG_CONFIDENCE,
                "ticker": TICKER_CONFIDENCE,
                "alias": ALIAS_CONFIDENCE,
            }[method]
            matches.append(
                EntityMatch(
                    entity_id=comp.ref.entity_id,
                    ticker=comp.ref.ticker,
                    confidence=confidence,
                    method=method,
                )
            )
        matches.sort(key=lambda m: (-m.confidence, m.ticker))
        return matches

    @staticmethod
    def _strongest_method(comp: _CompiledEntity, corpus: str) -> str | None:
        if comp.cashtag.search(corpus):
            return "cashtag"
        if comp.ticker is not None and comp.ticker.search(corpus):
            return "ticker"
        if any(pattern.search(corpus) for pattern in comp.aliases):
            return "alias"
        return None


def refs_from_rows(rows: Iterable[tuple[int, str, str, list[str] | None]]) -> list[EntityRef]:
    """Adapt (id, ticker, name, aliases) rows into EntityRefs (aliases may be NULL)."""
    return [
        EntityRef(entity_id=eid, ticker=ticker, name=name, aliases=tuple(aliases or ()))
        for eid, ticker, name, aliases in rows
    ]
