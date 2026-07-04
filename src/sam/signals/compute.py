"""Pure SAI math — deterministic functions, no I/O, no clock, no config.

Everything here maps plain inputs to plain outputs so the P5 gate
("deterministic rebuild from raw") reduces to: same rows in, same panel out.
The pipeline supplies repository rows and settings; this module never reads
either on its own.

Point-in-time decisions implemented here (rationale in
docs/sai_methodology.md):

- Documents are bucketed by the UTC day of ``ingested_at`` (*known* time).
  Closed days are immutable — new documents can only land "today" — which is
  what makes rebuilds reproducible and backtests leak-free.
- A staleness guard drops documents whose ``published_at`` predates ingestion
  by more than ``max_doc_age_days``: a late backfill is not an attention
  spike. Documents without ``published_at`` are treated as fresh.
- Topic velocity for day D uses the topic-model version *as of* D (latest
  fit created on or before the end of D). Versions are append-only, so this
  choice is itself reproducible.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any, NamedTuple

# Engagement keys summed into the engagement sub-signal. Hacker News writes
# score/comments today; future collectors (Reddit) must normalize onto these
# keys rather than growing this tuple per source.
ENGAGEMENT_KEYS = ("score", "comments")

# Component names — keys of the composite weights mapping.
MENTIONS = "mentions"
SENTIMENT = "sentiment"
TOPICS = "topics"
ENGAGEMENT = "engagement"


class LinkRow(NamedTuple):
    """One (document, entity) link with everything SAI needs from the doc."""

    document_id: int
    entity_id: int
    confidence: float
    ingested_at: datetime
    published_at: datetime | None
    engagement: dict[str, Any]
    sentiment: float | None  # signed, in [-1, 1]; None = unscored


class TopicRow(NamedTuple):
    """One document→topic assignment under one topic-model version."""

    document_id: int
    ingested_at: datetime
    published_at: datetime | None
    topic_id: int
    version: str
    probability: float


@dataclass(frozen=True, slots=True)
class DayAggregate:
    """Raw per-(entity, day) aggregates before any growth transform."""

    mentions: float = 0.0
    sentiment: float | None = None  # confidence-weighted mean; None = unscored
    engagement: float = 0.0


@dataclass(frozen=True, slots=True)
class PanelRow:
    """One computed sai_daily row (pre-persistence)."""

    entity_id: int
    day: date
    mention_growth: float | None
    sentiment_momentum: float | None
    topic_velocity: float | None
    engagement_growth: float | None
    sai_score: float | None
    sai_rank: int | None


def as_utc(value: datetime) -> datetime:
    """SQLite returns naive datetimes for tz-aware columns; pin to UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def utc_day(value: datetime) -> date:
    """The UTC calendar day a timestamp falls on."""
    return as_utc(value).date()


def is_stale(published_at: datetime | None, ingested_at: datetime, max_doc_age_days: int) -> bool:
    """True when the doc was published long before we learned of it.

    Stale docs are excluded from *all* day-aggregates — counting a backfilled
    old article as today's attention would fabricate a spike. No
    ``published_at`` means we cannot prove staleness: treated as fresh.
    """
    if published_at is None:
        return False
    return as_utc(ingested_at) - as_utc(published_at) > timedelta(days=max_doc_age_days)


def engagement_value(engagement: Mapping[str, Any]) -> float:
    """Sum the known numeric engagement keys (missing/None/non-numeric = 0)."""
    total = 0.0
    for key in ENGAGEMENT_KEYS:
        value = engagement.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total += float(value)
    return total


def daily_entity_aggregates(
    links: Iterable[LinkRow], *, max_doc_age_days: int
) -> dict[tuple[int, date], DayAggregate]:
    """Fold link rows into per-(entity, UTC day) raw aggregates.

    mentions = Σ link confidence (weight, don't filter — a low-confidence
    mention still counts, just less). sentiment = confidence-weighted mean of
    signed scores over the *scored* docs only (an unscored doc is missing
    data, not neutral tone). engagement = Σ confidence × engagement_value.
    """
    mentions: dict[tuple[int, date], float] = {}
    sent_num: dict[tuple[int, date], float] = {}
    sent_den: dict[tuple[int, date], float] = {}
    engagement: dict[tuple[int, date], float] = {}

    for row in links:
        if is_stale(row.published_at, row.ingested_at, max_doc_age_days):
            continue
        key = (row.entity_id, utc_day(row.ingested_at))
        mentions[key] = mentions.get(key, 0.0) + row.confidence
        engagement[key] = engagement.get(key, 0.0) + row.confidence * engagement_value(
            row.engagement
        )
        if row.sentiment is not None:
            sent_num[key] = sent_num.get(key, 0.0) + row.confidence * row.sentiment
            sent_den[key] = sent_den.get(key, 0.0) + row.confidence

    return {
        key: DayAggregate(
            mentions=mentions[key],
            sentiment=(sent_num[key] / sent_den[key]) if sent_den.get(key) else None,
            engagement=engagement[key],
        )
        for key in mentions
    }


def growth(
    series: Mapping[date, float],
    day: date,
    *,
    panel_start: date,
    window_days: int,
    min_history_days: int,
) -> float | None:
    """Day value vs its trailing-window mean: (v - mean) / max(mean, 1).

    Missing days count as 0.0 (a dense panel: no activity *is* data). The
    denominator floor keeps small-count growth bounded and defined when the
    baseline is zero. Returns None until the panel has ``min_history_days``
    of history — a growth rate against no baseline is noise, not signal.
    """
    if (day - panel_start).days < min_history_days:
        return None
    window = _window(day, panel_start, window_days)
    if not window:
        return None
    baseline = sum(series.get(d, 0.0) for d in window) / len(window)
    return (series.get(day, 0.0) - baseline) / max(baseline, 1.0)


def momentum(
    series: Mapping[date, float | None],
    day: date,
    *,
    panel_start: date,
    window_days: int,
    min_history_days: int,
) -> float | None:
    """Day value minus the trailing mean of *defined* values.

    Unlike :func:`growth`, missing days are gaps, not zeros — sentiment on a
    day with no scored documents is unknown. Requires the day itself to be
    defined and at least ``min_history_days`` defined values in the window.
    """
    if (day - panel_start).days < min_history_days:
        return None
    today = series.get(day)
    if today is None:
        return None
    defined = [
        v for d in _window(day, panel_start, window_days) if (v := series.get(d)) is not None
    ]
    if len(defined) < min_history_days:
        return None
    return today - sum(defined) / len(defined)


def _window(day: date, panel_start: date, window_days: int) -> list[date]:
    """Trailing lookback [day - window_days, day - 1], clipped to the panel."""
    first = max(panel_start, day - timedelta(days=window_days))
    return [first + timedelta(days=i) for i in range((day - first).days)]


def select_version_as_of(versions: Sequence[tuple[str, datetime]], day: date) -> str | None:
    """Latest topic-model version fitted on or before the end of ``day``.

    Point-in-time rule: a rebuild of a past day must use the topic model that
    existed then, not today's refit. ``versions`` is (version, created_at)
    oldest-first (see TopicRepository.versions); returns None before the
    first fit.
    """
    end_of_day = datetime.combine(day + timedelta(days=1), datetime.min.time(), tzinfo=UTC)
    chosen: str | None = None
    for version, created_at in versions:
        if as_utc(created_at) < end_of_day:
            chosen = version
    return chosen


def topic_day_weights(
    topic_rows: Iterable[TopicRow], *, max_doc_age_days: int
) -> dict[str, dict[tuple[int, date], float]]:
    """Per version: probability-weighted doc count per (topic, UTC day).

    The global "how big is this narrative today" series that topic velocity
    measures growth against. Same staleness guard as the entity aggregates.
    """
    totals: dict[str, dict[tuple[int, date], float]] = {}
    for row in topic_rows:
        if is_stale(row.published_at, row.ingested_at, max_doc_age_days):
            continue
        key = (row.topic_id, utc_day(row.ingested_at))
        per_version = totals.setdefault(row.version, {})
        per_version[key] = per_version.get(key, 0.0) + row.probability
    return totals


def first_activity_day(links: Iterable[LinkRow], *, max_doc_age_days: int) -> date | None:
    """UTC day of the earliest fresh linked document — the panel's origin.

    The history gate measures from here; stale docs don't move it (they never
    enter any aggregate).
    """
    days = [
        utc_day(row.ingested_at)
        for row in links
        if not is_stale(row.published_at, row.ingested_at, max_doc_age_days)
    ]
    return min(days) if days else None


def centered_ranks(values: Mapping[int, float]) -> dict[int, float]:
    """Cross-sectional average ranks rescaled to [-1, 1] (ties share ranks).

    Scale-free and robust at small n — and the eventual validation metric
    (IC) is rank correlation anyway, so ranking here loses nothing the
    kill-gate would measure. A single entity centers to 0.0.
    """
    n = len(values)
    if n == 0:
        return {}
    if n == 1:
        return {next(iter(values)): 0.0}
    ordered = sorted(values.items(), key=lambda kv: (kv[1], kv[0]))
    ranks: dict[int, float] = {}
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ordered[j + 1][1] == ordered[i][1]:
            j += 1
        shared = (i + j) / 2 + 1  # mean of 1-based ranks i+1 .. j+1
        for k in range(i, j + 1):
            ranks[ordered[k][0]] = shared
        i = j + 1
    half_span = (n - 1) / 2
    return {entity: (rank - (n + 1) / 2) / half_span for entity, rank in ranks.items()}


def composite_scores(
    components: Mapping[str, Mapping[int, float | None]],
    weights: Mapping[str, float],
    entity_ids: Iterable[int],
) -> dict[int, float | None]:
    """Weighted mean of per-component centered ranks; None-safe.

    Components an entity lacks (insufficient history) drop out and their
    weight is renormalized away — a young panel yields a score from whatever
    is measurable rather than dragging absent components in as zeros. An
    entity with no measurable component gets None.
    """
    ranked = {
        name: centered_ranks({e: v for e, v in comp.items() if v is not None})
        for name, comp in components.items()
    }
    scores: dict[int, float | None] = {}
    for entity_id in entity_ids:
        numerator = denominator = 0.0
        for name, ranks in ranked.items():
            if entity_id in ranks:
                weight = weights.get(name, 0.0)
                numerator += weight * ranks[entity_id]
                denominator += weight
        scores[entity_id] = numerator / denominator if denominator > 0 else None
    return scores


def sai_ranks(scores: Mapping[int, float | None]) -> dict[int, int | None]:
    """1 = strongest score of the day; ties broken by entity id (stable)."""
    scored = sorted(
        ((e, s) for e, s in scores.items() if s is not None), key=lambda kv: (-kv[1], kv[0])
    )
    ranks: dict[int, int | None] = dict.fromkeys(scores)
    for position, (entity_id, _score) in enumerate(scored, start=1):
        ranks[entity_id] = position
    return ranks


def build_panel(
    links: Sequence[LinkRow],
    topic_rows: Sequence[TopicRow],
    versions: Sequence[tuple[str, datetime]],
    entity_ids: Sequence[int],
    days: Sequence[date],
    *,
    panel_start: date,
    window_days: int,
    min_history_days: int,
    max_doc_age_days: int,
    weights: Mapping[str, float],
) -> list[PanelRow]:
    """Compute the dense SAI panel for ``entity_ids`` × ``days``.

    Pure: the caller supplies every input (rows, the day range, the panel
    origin for history gating, and settings values). Emits one row per
    entity per day — zero-activity days are real observations (an attention
    collapse is signal), and NULL components mean insufficient history.
    """
    aggregates = daily_entity_aggregates(links, max_doc_age_days=max_doc_age_days)
    mention_series: dict[int, dict[date, float]] = {e: {} for e in entity_ids}
    sentiment_series: dict[int, dict[date, float | None]] = {e: {} for e in entity_ids}
    engagement_series: dict[int, dict[date, float]] = {e: {} for e in entity_ids}
    for (entity_id, day), agg in aggregates.items():
        if entity_id not in mention_series:
            continue  # link to an entity outside the requested panel
        mention_series[entity_id][day] = agg.mentions
        sentiment_series[entity_id][day] = agg.sentiment
        engagement_series[entity_id][day] = agg.engagement

    topic_totals = topic_day_weights(topic_rows, max_doc_age_days=max_doc_age_days)
    doc_topics = _doc_topics_by_version(topic_rows)
    fresh_links_by_day = _fresh_links_by_day(links, max_doc_age_days)

    rows: list[PanelRow] = []
    for day in days:
        mention_growth = {
            e: growth(
                mention_series[e],
                day,
                panel_start=panel_start,
                window_days=window_days,
                min_history_days=min_history_days,
            )
            for e in entity_ids
        }
        sentiment_momentum = {
            e: momentum(
                sentiment_series[e],
                day,
                panel_start=panel_start,
                window_days=window_days,
                min_history_days=min_history_days,
            )
            for e in entity_ids
        }
        engagement_growth = {
            e: growth(
                engagement_series[e],
                day,
                panel_start=panel_start,
                window_days=window_days,
                min_history_days=min_history_days,
            )
            for e in entity_ids
        }
        topic_velocity = _topic_velocity_for_day(
            day,
            entity_ids=entity_ids,
            versions=versions,
            topic_totals=topic_totals,
            doc_topics=doc_topics,
            fresh_links=fresh_links_by_day.get(day, []),
            panel_start=panel_start,
            window_days=window_days,
            min_history_days=min_history_days,
        )

        components: dict[str, Mapping[int, float | None]] = {
            MENTIONS: mention_growth,
            SENTIMENT: sentiment_momentum,
            TOPICS: topic_velocity,
            ENGAGEMENT: engagement_growth,
        }
        scores = composite_scores(components, weights, entity_ids)
        ranks = sai_ranks(scores)
        rows.extend(
            PanelRow(
                entity_id=e,
                day=day,
                mention_growth=mention_growth[e],
                sentiment_momentum=sentiment_momentum[e],
                topic_velocity=topic_velocity[e],
                engagement_growth=engagement_growth[e],
                sai_score=scores[e],
                sai_rank=ranks[e],
            )
            for e in entity_ids
        )
    return rows


def _doc_topics_by_version(
    topic_rows: Iterable[TopicRow],
) -> dict[str, dict[int, list[tuple[int, float]]]]:
    """version -> document_id -> [(topic_id, probability)]."""
    result: dict[str, dict[int, list[tuple[int, float]]]] = {}
    for row in topic_rows:
        result.setdefault(row.version, {}).setdefault(row.document_id, []).append(
            (row.topic_id, row.probability)
        )
    return result


def _fresh_links_by_day(
    links: Iterable[LinkRow], max_doc_age_days: int
) -> dict[date, list[LinkRow]]:
    by_day: dict[date, list[LinkRow]] = {}
    for row in links:
        if is_stale(row.published_at, row.ingested_at, max_doc_age_days):
            continue
        by_day.setdefault(utc_day(row.ingested_at), []).append(row)
    return by_day


def _topic_velocity_for_day(
    day: date,
    *,
    entity_ids: Sequence[int],
    versions: Sequence[tuple[str, datetime]],
    topic_totals: Mapping[str, Mapping[tuple[int, date], float]],
    doc_topics: Mapping[str, Mapping[int, list[tuple[int, float]]]],
    fresh_links: Sequence[LinkRow],
    panel_start: date,
    window_days: int,
    min_history_days: int,
) -> dict[int, float | None]:
    """Entity's weighted mean of its day-D topics' growth rates.

    Weight per (entity, topic) = link confidence × assignment probability,
    summed over the entity's fresh day-D documents. Topic growth reuses
    :func:`growth` over the version-scoped global topic series — "the
    narratives this entity sits in today are accelerating".
    """
    velocities: dict[int, float | None] = dict.fromkeys(entity_ids)
    version = select_version_as_of(versions, day)
    if version is None or (day - panel_start).days < min_history_days:
        return velocities

    totals = topic_totals.get(version, {})
    assignments = doc_topics.get(version, {})
    topic_growth_memo: dict[int, float | None] = {}

    def topic_growth(topic_id: int) -> float | None:
        if topic_id not in topic_growth_memo:
            series = {d: value for (t, d), value in totals.items() if t == topic_id}
            topic_growth_memo[topic_id] = growth(
                series,
                day,
                panel_start=panel_start,
                window_days=window_days,
                min_history_days=min_history_days,
            )
        return topic_growth_memo[topic_id]

    numerator: dict[int, float] = {}
    denominator: dict[int, float] = {}
    for link in fresh_links:
        for topic_id, probability in assignments.get(link.document_id, []):
            rate = topic_growth(topic_id)
            if rate is None:
                continue
            weight = link.confidence * probability
            numerator[link.entity_id] = numerator.get(link.entity_id, 0.0) + weight * rate
            denominator[link.entity_id] = denominator.get(link.entity_id, 0.0) + weight

    for entity_id in entity_ids:
        if denominator.get(entity_id):
            velocities[entity_id] = numerator[entity_id] / denominator[entity_id]
    return velocities
