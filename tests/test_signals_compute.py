"""Pure SAI-math tests (Phase 5) — every expected value is computed by hand.

No DB, no config, no clock: sam.signals.compute maps plain inputs to plain
outputs, so these tests are the ground truth for the deterministic-rebuild
gate. The pipeline tests (test_signals_pipeline.py) only verify plumbing.
"""

from __future__ import annotations

from datetime import UTC, date, datetime

import pytest

from sam.signals.compute import (
    DayAggregate,
    LinkRow,
    TopicRow,
    build_panel,
    centered_ranks,
    composite_scores,
    daily_entity_aggregates,
    engagement_value,
    growth,
    is_stale,
    momentum,
    sai_ranks,
    select_version_as_of,
    topic_day_weights,
    utc_day,
)

D = date(2026, 7, 10)  # an arbitrary panel day


def _ts(day: date, hour: int = 12) -> datetime:
    return datetime(day.year, day.month, day.day, hour, tzinfo=UTC)


def _link(
    entity_id: int = 1,
    day: date = D,
    *,
    doc_id: int = 1,
    confidence: float = 1.0,
    sentiment: float | None = None,
    engagement: dict[str, object] | None = None,
    published_at: datetime | None = None,
) -> LinkRow:
    return LinkRow(
        document_id=doc_id,
        entity_id=entity_id,
        confidence=confidence,
        ingested_at=_ts(day),
        published_at=published_at if published_at is not None else _ts(day, hour=10),
        engagement=engagement or {},
        sentiment=sentiment,
    )


# ------------------------------------------------------------- foundations


def test_utc_day_handles_naive_and_aware() -> None:
    aware = datetime(2026, 7, 3, 23, 30, tzinfo=UTC)
    naive = datetime(2026, 7, 3, 23, 30)  # SQLite round-trip loses tzinfo
    assert utc_day(aware) == date(2026, 7, 3)
    assert utc_day(naive) == date(2026, 7, 3)
    # A non-UTC timestamp converts before bucketing (01:30+02:00 = 23:30 UTC prev day).
    from datetime import timedelta, timezone

    cest = timezone(timedelta(hours=2))
    assert utc_day(datetime(2026, 7, 4, 1, 30, tzinfo=cest)) == date(2026, 7, 3)


def test_is_stale_guards_backfills_but_trusts_missing_published_at() -> None:
    ingested = _ts(D)
    assert not is_stale(_ts(D, hour=1), ingested, max_doc_age_days=7)
    assert is_stale(datetime(2026, 6, 1, tzinfo=UTC), ingested, max_doc_age_days=7)
    assert not is_stale(None, ingested, max_doc_age_days=7)  # unknown = fresh


def test_engagement_value_sums_known_numeric_keys_only() -> None:
    assert engagement_value({"score": 10, "comments": 2}) == 12.0
    assert engagement_value({"feed": "CNBC"}) == 0.0  # RSS carries no numbers
    assert engagement_value({"score": None, "comments": 3}) == 3.0
    assert engagement_value({"score": True}) == 0.0  # bools are not counts
    assert engagement_value({}) == 0.0


# --------------------------------------------------------------- aggregates


def test_daily_entity_aggregates_weight_by_confidence() -> None:
    links = [
        _link(doc_id=1, confidence=1.0, sentiment=0.8, engagement={"score": 10}),
        _link(doc_id=2, confidence=0.5, sentiment=-0.4, engagement={"score": 4, "comments": 2}),
        _link(doc_id=3, confidence=0.5, sentiment=None),  # unscored: mentions yes, tone no
    ]
    (agg,) = daily_entity_aggregates(links, max_doc_age_days=7).values()
    assert agg.mentions == 2.0  # 1.0 + 0.5 + 0.5
    # (1.0*0.8 + 0.5*-0.4) / (1.0 + 0.5) = 0.6 / 1.5 — doc 3 not in denominator
    assert agg.sentiment == pytest.approx(0.4)
    assert agg.engagement == pytest.approx(1.0 * 10 + 0.5 * 6)  # 13.0


def test_daily_entity_aggregates_drop_stale_docs_and_split_days() -> None:
    stale = _link(doc_id=1, published_at=datetime(2026, 1, 1, tzinfo=UTC))
    fresh_today = _link(doc_id=2)
    fresh_yesterday = _link(doc_id=3, day=date(2026, 7, 9))
    aggs = daily_entity_aggregates([stale, fresh_today, fresh_yesterday], max_doc_age_days=7)
    assert set(aggs) == {(1, D), (1, date(2026, 7, 9))}
    assert aggs[(1, D)].mentions == 1.0  # the stale doc never counted


def test_day_aggregate_unscored_day_has_none_sentiment() -> None:
    (agg,) = daily_entity_aggregates([_link(sentiment=None)], max_doc_age_days=7).values()
    assert agg == DayAggregate(mentions=1.0, sentiment=None, engagement=0.0)


# ------------------------------------------------------- growth & momentum


def test_growth_vs_trailing_mean_with_zero_fill() -> None:
    start = date(2026, 7, 1)
    series = {date(2026, 7, 7): 4.0, date(2026, 7, 8): 2.0, D: 9.0}
    # Window 7/03..7/09 -> values (0,0,0,0,4,2,0), mean 6/7; floor max(6/7,1)=1.
    value = growth(series, D, panel_start=start, window_days=7, min_history_days=3)
    assert value == pytest.approx((9.0 - 6 / 7) / 1.0)


def test_growth_baseline_floor_bounds_small_count_noise() -> None:
    start = date(2026, 7, 1)
    series = {date(2026, 7, 8): 3.0, date(2026, 7, 9): 3.0, D: 6.0}
    # Window mean = 6/7 < 1 would explode a ratio; the floor caps it... but a
    # baseline above 1 divides normally: window (0,0,0,0,0,3,3), mean 6/7 -> floor.
    floored = growth(series, D, panel_start=start, window_days=7, min_history_days=3)
    assert floored == pytest.approx(6.0 - 6 / 7)
    # With a real baseline (mean 3), division is by the mean itself.
    dense = dict.fromkeys((date(2026, 7, i) for i in range(3, 10)), 3.0) | {D: 6.0}
    assert growth(dense, D, panel_start=start, window_days=7, min_history_days=3) == pytest.approx(
        (6.0 - 3.0) / 3.0
    )


def test_growth_returns_none_before_min_history() -> None:
    start = date(2026, 7, 9)
    series = {start: 5.0, D: 9.0}
    assert growth(series, D, panel_start=start, window_days=7, min_history_days=3) is None
    # Exactly at the gate (3 days after panel start) it computes.
    later = date(2026, 7, 12)
    assert growth(series, later, panel_start=start, window_days=7, min_history_days=3) is not None


def test_momentum_ignores_gaps_and_requires_defined_today() -> None:
    start = date(2026, 7, 1)
    series: dict[date, float | None] = {
        date(2026, 7, 6): 0.2,
        date(2026, 7, 7): None,  # mentioned but unscored day
        date(2026, 7, 8): 0.4,
        date(2026, 7, 9): 0.6,
        D: 0.9,
    }
    # Defined window values: 0.2, 0.4, 0.6 -> mean 0.4; momentum 0.5. Gaps skipped.
    value = momentum(series, D, panel_start=start, window_days=7, min_history_days=3)
    assert value == pytest.approx(0.5)
    # Today undefined -> None even with a rich window.
    series[D] = None
    assert momentum(series, D, panel_start=start, window_days=7, min_history_days=3) is None
    # Fewer defined window values than min_history -> None.
    sparse: dict[date, float | None] = {date(2026, 7, 9): 0.5, D: 0.9}
    assert momentum(sparse, D, panel_start=start, window_days=7, min_history_days=3) is None


# ------------------------------------------------------------------ topics


def test_select_version_as_of_is_point_in_time() -> None:
    versions = [
        ("v1", datetime(2026, 7, 5, 8, tzinfo=UTC)),
        ("v2", datetime(2026, 7, 9, 8, tzinfo=UTC)),
    ]
    assert select_version_as_of(versions, date(2026, 7, 4)) is None  # before any fit
    assert select_version_as_of(versions, date(2026, 7, 5)) == "v1"  # fit that day counts
    assert select_version_as_of(versions, date(2026, 7, 8)) == "v1"  # v2 not yet known
    assert select_version_as_of(versions, D) == "v2"
    assert select_version_as_of([], D) is None


def test_topic_day_weights_sum_probabilities_per_version() -> None:
    rows = [
        TopicRow(1, _ts(D), _ts(D, 10), topic_id=7, version="v1", probability=0.8),
        TopicRow(2, _ts(D), _ts(D, 10), topic_id=7, version="v1", probability=0.4),
        TopicRow(1, _ts(D), _ts(D, 10), topic_id=9, version="v2", probability=0.5),
        # Stale doc excluded everywhere.
        TopicRow(3, _ts(D), datetime(2026, 1, 1, tzinfo=UTC), 7, "v1", 1.0),
    ]
    totals = topic_day_weights(rows, max_doc_age_days=7)
    assert totals == {"v1": {(7, D): pytest.approx(1.2)}, "v2": {(9, D): 0.5}}


# ------------------------------------------------------ ranks & composite


def test_centered_ranks_span_minus_one_to_one() -> None:
    assert centered_ranks({1: 0.1, 2: 0.5, 3: 0.9}) == {1: -1.0, 2: 0.0, 3: 1.0}
    assert centered_ranks({42: 3.0}) == {42: 0.0}  # lone entity centers to zero
    assert centered_ranks({}) == {}


def test_centered_ranks_average_ties() -> None:
    # Values 1, 5, 5, 9 -> ranks 1, 2.5, 2.5, 4 -> centered (r-2.5)/1.5.
    ranks = centered_ranks({1: 1.0, 2: 5.0, 3: 5.0, 4: 9.0})
    assert ranks == {1: -1.0, 2: 0.0, 3: 0.0, 4: 1.0}


def test_composite_renormalizes_weights_over_missing_components() -> None:
    components = {
        "mentions": {1: 2.0, 2: -1.0},
        "sentiment": {1: None, 2: None},  # nobody has history yet
        "topics": {1: None, 2: 0.5},
        "engagement": {1: 0.0, 2: 4.0},
    }
    weights = {"mentions": 0.25, "sentiment": 0.25, "topics": 0.25, "engagement": 0.25}
    scores = composite_scores(components, weights, entity_ids=[1, 2, 3])
    # Entity 1: mentions rank +1, engagement rank -1 -> (0.25*1 + 0.25*-1)/0.5 = 0.
    assert scores[1] == pytest.approx(0.0)
    # Entity 2: mentions -1, topics 0 (lone), engagement +1 -> 0/0.75 = 0.
    assert scores[2] == pytest.approx(0.0)
    assert scores[3] is None  # not in any component


def test_composite_weights_change_the_blend() -> None:
    components = {
        "mentions": {1: 2.0, 2: -1.0},
        "engagement": {1: -3.0, 2: 5.0},
    }
    weights = {"mentions": 0.75, "engagement": 0.25}
    scores = composite_scores(components, weights, entity_ids=[1, 2])
    # Entity 1: 0.75*(+1) + 0.25*(-1) = 0.5 over total weight 1.0.
    assert scores[1] == pytest.approx(0.5)
    assert scores[2] == pytest.approx(-0.5)


def test_sai_ranks_order_and_tie_break() -> None:
    ranks = sai_ranks({1: 0.5, 2: 0.9, 3: None, 4: 0.5})
    assert ranks == {2: 1, 1: 2, 4: 3, 3: None}  # tie 1 vs 4 -> lower id first


# ------------------------------------------------------------- full panel


def test_build_panel_gates_history_then_emits_growth() -> None:
    start = date(2026, 7, 1)
    days = [date(2026, 7, 1), date(2026, 7, 2), date(2026, 7, 4)]
    links = [
        _link(entity_id=1, day=date(2026, 7, 1), doc_id=1, sentiment=0.5),
        _link(entity_id=1, day=date(2026, 7, 2), doc_id=2, sentiment=0.5),
        _link(entity_id=1, day=date(2026, 7, 3), doc_id=3, sentiment=0.5),
        _link(entity_id=1, day=date(2026, 7, 4), doc_id=4, confidence=1.0, sentiment=0.9),
        _link(entity_id=2, day=date(2026, 7, 4), doc_id=5, confidence=0.5, sentiment=None),
    ]
    rows = build_panel(
        links,
        topic_rows=[],
        versions=[],
        entity_ids=[1, 2],
        days=days,
        panel_start=start,
        window_days=7,
        min_history_days=3,
        max_doc_age_days=7,
        weights={"mentions": 0.25, "sentiment": 0.25, "topics": 0.25, "engagement": 0.25},
    )
    by_key = {(r.entity_id, r.day): r for r in rows}
    assert len(rows) == 6  # dense: 2 entities x 3 days

    # Days 1-2 are inside the min-history gate: everything NULL.
    assert by_key[(1, date(2026, 7, 1))].mention_growth is None
    assert by_key[(1, date(2026, 7, 2))].sai_score is None

    # Day 4 (3 days after panel start): entity 1 window 7/01..7/03 mean = 1.0.
    day4_e1 = by_key[(1, date(2026, 7, 4))]
    assert day4_e1.mention_growth == pytest.approx((1.0 - 1.0) / 1.0)  # flat
    # Sentiment window: 0.5, 0.5, 0.5 -> mean 0.5; today 0.9 -> momentum 0.4.
    assert day4_e1.sentiment_momentum == pytest.approx(0.4)
    assert day4_e1.topic_velocity is None  # no topic model fitted

    # Entity 2 first appears on day 4: mentions 0.5 vs zero baseline.
    day4_e2 = by_key[(2, date(2026, 7, 4))]
    assert day4_e2.mention_growth == pytest.approx(0.5)  # (0.5 - 0) / max(0, 1)
    assert day4_e2.sentiment_momentum is None  # unscored doc

    # Cross-sectional ranks: e2 mention growth (0.5) > e1 (0.0) -> e2 ranks first.
    assert day4_e2.sai_score is not None and day4_e1.sai_score is not None
    assert day4_e2.sai_score > day4_e1.sai_score
    assert (day4_e2.sai_rank, day4_e1.sai_rank) == (1, 2)


def test_build_panel_topic_velocity_uses_as_of_version() -> None:
    start = date(2026, 7, 1)
    fit_day = datetime(2026, 7, 8, 6, tzinfo=UTC)
    versions = [("v1", fit_day)]
    # Topic 7 counts: 1.0/day on 7/07-7/09, then 3.0 on 7/10 (acceleration).
    topic_rows = [
        TopicRow(doc, _ts(day), _ts(day, 10), topic_id=7, version="v1", probability=1.0)
        for doc, day in [
            (1, date(2026, 7, 7)),
            (2, date(2026, 7, 8)),
            (3, date(2026, 7, 9)),
            (4, D),
            (5, D),
            (6, D),
        ]
    ]
    links = [_link(entity_id=1, day=D, doc_id=4, confidence=0.8)]
    rows = build_panel(
        links,
        topic_rows=topic_rows,
        versions=versions,
        entity_ids=[1],
        days=[date(2026, 7, 7), D],
        panel_start=start,
        window_days=7,
        min_history_days=3,
        max_doc_age_days=7,
        weights={"mentions": 0.25, "sentiment": 0.25, "topics": 0.25, "engagement": 0.25},
    )
    by_day = {r.day: r for r in rows}
    # 7/07 predates the fit: no version as of that day -> NULL velocity.
    assert by_day[date(2026, 7, 7)].topic_velocity is None
    # 7/10: topic-7 window 7/03..7/09 -> (0,0,0,0,1,1,1), mean 3/7; today 3.0.
    assert by_day[D].topic_velocity == pytest.approx((3.0 - 3 / 7) / 1.0)
