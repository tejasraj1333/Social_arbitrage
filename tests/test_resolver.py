"""Unit tests for the pure text→ticker matcher (Phase 3 / M2).

The matcher is DB-free, so every rule tier, boundary case, and ambiguity
guard is exercised directly.
"""

from __future__ import annotations

import unicodedata

import pytest

from sam.processing.resolver import (
    ALIAS_CONFIDENCE,
    CASHTAG_CONFIDENCE,
    TICKER_CONFIDENCE,
    EntityMatch,
    EntityMatcher,
    EntityRef,
    refs_from_rows,
)

NVDA = EntityRef(entity_id=1, ticker="NVDA", name="NVIDIA Corporation", aliases=("Nvidia",))
AAPL = EntityRef(entity_id=2, ticker="AAPL", name="Apple Inc.", aliases=("Apple",))
AMZN = EntityRef(entity_id=3, ticker="AMZN", name="Amazon.com, Inc.", aliases=("Amazon",))
ALL_ = EntityRef(entity_id=4, ticker="ALL", name="Allstate Corporation", aliases=("Allstate",))


@pytest.fixture
def matcher() -> EntityMatcher:
    return EntityMatcher([NVDA, AAPL, AMZN, ALL_])


def test_cashtag_match_is_strongest(matcher: EntityMatcher) -> None:
    (match,) = matcher.match("Retail piles into $NVDA ahead of earnings")
    assert match == EntityMatch(1, "NVDA", CASHTAG_CONFIDENCE, "cashtag")


def test_cashtag_is_case_insensitive(matcher: EntityMatcher) -> None:
    (match,) = matcher.match("loading up on $nvda calls")
    assert match.method == "cashtag"


def test_bare_ticker_matches_uppercase_only(matcher: EntityMatcher) -> None:
    (match,) = matcher.match("NVDA closed up 4% on record data-center revenue")
    assert match == EntityMatch(1, "NVDA", TICKER_CONFIDENCE, "ticker")
    assert matcher.match("nvda closed up 4%") == []  # lowercase prose = noise


def test_ticker_respects_word_boundaries(matcher: EntityMatcher) -> None:
    assert matcher.match("the ANVDA conference") == []
    assert matcher.match("NVDAX fund holdings") == []
    (match,) = matcher.match("NASDAQ:NVDA hits a new high")  # ':' is a boundary
    assert match.ticker == "NVDA"


def test_alias_match_case_insensitive(matcher: EntityMatcher) -> None:
    (match,) = matcher.match("nvidia unveils its next-gen GPU lineup")
    assert match == EntityMatch(1, "NVDA", ALIAS_CONFIDENCE, "alias")


def test_full_company_name_matches_even_with_punctuation(matcher: EntityMatcher) -> None:
    # "Amazon.com, Inc." ends in a non-word char — the lookaround boundaries
    # must still match it at end of sentence.
    matches = matcher.match("Quarterly filing by Amazon.com, Inc.")
    assert [m.ticker for m in matches] == ["AMZN"]


def test_strongest_rule_wins_per_entity(matcher: EntityMatcher) -> None:
    (match,) = matcher.match("$NVDA is mooning — Nvidia can do no wrong")
    assert match.method == "cashtag"
    assert match.confidence == CASHTAG_CONFIDENCE


def test_ambiguous_ticker_needs_cashtag_or_alias(matcher: EntityMatcher) -> None:
    # "ALL" is in AMBIGUOUS_TICKERS: bare token must NOT match...
    assert matcher.match("We ALL know how this ends") == []
    # ...but the cashtag and the alias still do.
    (via_cashtag,) = matcher.match("$ALL printed a nice quarter")
    assert via_cashtag.method == "cashtag"
    (via_alias,) = matcher.match("Allstate raises its dividend")
    assert via_alias == EntityMatch(4, "ALL", ALIAS_CONFIDENCE, "alias")


def test_multiple_entities_sorted_by_confidence_then_ticker(matcher: EntityMatcher) -> None:
    matches = matcher.match("Apple and $NVDA both rallied; Amazon lagged")
    assert [(m.ticker, m.method) for m in matches] == [
        ("NVDA", "cashtag"),
        ("AAPL", "alias"),
        ("AMZN", "alias"),
    ]


def test_matches_across_multiple_text_fields(matcher: EntityMatcher) -> None:
    matches = matcher.match("Chipmakers extend gains", "Nvidia led the sector higher")
    assert [m.ticker for m in matches] == ["NVDA"]


def test_none_and_empty_texts_yield_no_matches(matcher: EntityMatcher) -> None:
    assert matcher.match(None) == []
    assert matcher.match("", None) == []
    assert matcher.match() == []


def test_no_boundary_bleed_between_fields(matcher: EntityMatcher) -> None:
    # Fields are joined with separators, so a ticker split across two fields
    # must not produce a match.
    assert matcher.match("NV", "DA") == []


def test_refs_from_rows_tolerates_null_aliases() -> None:
    refs = refs_from_rows([(1, "NVDA", "NVIDIA Corporation", None)])
    assert refs == [EntityRef(1, "NVDA", "NVIDIA Corporation", ())]


def test_unicode_text_is_normalized_before_matching() -> None:
    # Alias compiled from the NFC form; document text arrives NFD-decomposed
    # (e + combining acute U+0301). match() must reconcile them via NFC.
    nfc_name = unicodedata.normalize("NFC", "Cafe" + chr(0x301) + " Holdings")
    nfd_text = unicodedata.normalize("NFD", nfc_name + " beats estimates")
    cafe = EntityRef(entity_id=9, ticker="CAFE", name=nfc_name, aliases=())
    matcher = EntityMatcher([cafe])
    (match,) = matcher.match(nfd_text)
    assert match.ticker == "CAFE"
    assert match.method == "alias"
