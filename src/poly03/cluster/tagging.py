"""§4.3: "500 independent-looking 'No' bets on Trump-adjacent political
markets are one bet on one political regime." This module tags each
market with the dimensions that can silently correlate a whole book
(entity, theme, geography, resolution source, resolution date bucket),
and tracks live exposure against the caps so the exclusion filter can ask
"would this trip a cluster cap?" before entering.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from urllib.parse import urlparse

from poly03.config import (
    DATE_BUCKET_WINDOW_DAYS,
    MAX_DATE_BUCKET_FRACTION,
    MAX_ENTITY_CLUSTER_FRACTION,
    MAX_RESOLUTION_SOURCE_FRACTION,
    MAX_THEME_CLUSTER_FRACTION,
)
from poly03.data.models import Market

logger = logging.getLogger("poly03.cluster")

_LEADING_STOPWORDS = {"Will", "The", "Is", "Does", "Are", "Was", "In", "A", "An", "Do"}

_PROPER_NOUN_RUN = re.compile(r"\b([A-Z][a-zA-Z.]*(?:\s+[A-Z][a-zA-Z.]*)*)\b")

_GEOGRAPHY_ALLOWLIST = (
    "United States", "China", "Russia", "Ukraine", "Iran", "Israel", "Gaza",
    "Ukraine", "Taiwan", "North Korea", "South Korea", "India", "Pakistan",
    "United Kingdom", "France", "Germany", "Mexico", "Cuba", "Venezuela",
    "Brazil", "Japan", "World", "Europe", "Middle East", "Africa", "Asia",
)


@dataclass(frozen=True)
class ClusterTags:
    market_id: str
    entity: str
    themes: tuple[str, ...]
    geography: str | None
    resolution_source: str
    date_bucket: str | None


def _extract_entity(market: Market) -> str:
    """Best-effort proper-noun extraction. group_item_title (Gamma's field
    for 'which candidate/outcome within a multi-outcome event') is the
    reliable case; otherwise fall back to the first capitalized run in the
    question that isn't a leading stopword."""
    if market.group_item_title:
        return market.group_item_title
    for match in _PROPER_NOUN_RUN.finditer(market.question):
        candidate = match.group(1)
        if candidate not in _LEADING_STOPWORDS and len(candidate) > 1:
            return candidate
    return market.question[:40]


def _extract_geography(market: Market) -> str | None:
    for g in _GEOGRAPHY_ALLOWLIST:
        if g in market.tags:
            return g
    text = market.question
    for g in _GEOGRAPHY_ALLOWLIST:
        if g in text:
            return g
    return None


def _normalize_resolution_source(market: Market) -> str:
    src = market.resolution_source.strip()
    if src:
        parsed = urlparse(src if "://" in src else f"//{src}")
        return parsed.netloc or src
    url_match = re.search(r"https?://([^\s/]+)", market.description)
    if url_match:
        return url_match.group(1)
    return "unknown"


def _date_bucket(market: Market, *, window_days: int = DATE_BUCKET_WINDOW_DAYS) -> str | None:
    if market.end_date is None:
        return None
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    days_since_epoch = (market.end_date - epoch).days
    bucket_index = days_since_epoch // window_days
    bucket_start = epoch + timedelta(days=bucket_index * window_days)
    return bucket_start.date().isoformat()


def tag_market(market: Market) -> ClusterTags:
    return ClusterTags(
        market_id=market.id,
        entity=_extract_entity(market),
        themes=tuple(market.tags) if market.tags else (),
        geography=_extract_geography(market),
        resolution_source=_normalize_resolution_source(market),
        date_bucket=_date_bucket(market),
    )


@dataclass
class ClusterExposureTracker:
    """Live $ exposure per cluster dimension, checked against §4.3 caps.
    bankroll is mutable on purpose -- call update_bankroll() as it changes
    so caps stay relative to current capital, not the value at construction.

    The four `*_cap_fraction` fields default to the module-level §4.3
    constants (Phase 0's behavior, unchanged) but can be overridden per
    instance -- execution.py's live engine passes the much larger
    MAKING_LIVE_MAX_*_CLUSTER_FRACTION knobs, since the Phase 0 fractions are
    sized for a bankroll orders of magnitude bigger than a real live
    bankroll_cap and would block every market's minimum quote on their own
    (task 20260818_2012 item 1b).
    """

    bankroll: float
    entity_exposure: dict[str, float] = field(default_factory=dict)
    theme_exposure: dict[str, float] = field(default_factory=dict)
    date_bucket_exposure: dict[str, float] = field(default_factory=dict)
    source_exposure: dict[str, float] = field(default_factory=dict)
    entity_cap_fraction: float = MAX_ENTITY_CLUSTER_FRACTION
    theme_cap_fraction: float = MAX_THEME_CLUSTER_FRACTION
    date_bucket_cap_fraction: float = MAX_DATE_BUCKET_FRACTION
    source_cap_fraction: float = MAX_RESOLUTION_SOURCE_FRACTION

    def update_bankroll(self, bankroll: float) -> None:
        self.bankroll = bankroll

    def would_breach(self, tags: ClusterTags, stake: float) -> list[str]:
        reasons = []
        if self.bankroll <= 0:
            return reasons

        entity_cap = self.entity_cap_fraction * self.bankroll
        if self.entity_exposure.get(tags.entity, 0.0) + stake > entity_cap:
            reasons.append(f"entity cluster '{tags.entity}' would exceed {self.entity_cap_fraction:.0%} cap")

        for theme in tags.themes:
            theme_cap = self.theme_cap_fraction * self.bankroll
            if self.theme_exposure.get(theme, 0.0) + stake > theme_cap:
                reasons.append(f"theme cluster '{theme}' would exceed {self.theme_cap_fraction:.0%} cap")

        if tags.date_bucket is not None:
            bucket_cap = self.date_bucket_cap_fraction * self.bankroll
            if self.date_bucket_exposure.get(tags.date_bucket, 0.0) + stake > bucket_cap:
                reasons.append(f"date bucket '{tags.date_bucket}' would exceed {self.date_bucket_cap_fraction:.0%} cap")

        source_cap = self.source_cap_fraction * self.bankroll
        if self.source_exposure.get(tags.resolution_source, 0.0) + stake > source_cap:
            reasons.append(
                f"resolution source '{tags.resolution_source}' would exceed {self.source_cap_fraction:.0%} cap"
            )

        return reasons

    def register(self, tags: ClusterTags, stake: float) -> None:
        self.entity_exposure[tags.entity] = self.entity_exposure.get(tags.entity, 0.0) + stake
        for theme in tags.themes:
            self.theme_exposure[theme] = self.theme_exposure.get(theme, 0.0) + stake
        if tags.date_bucket is not None:
            self.date_bucket_exposure[tags.date_bucket] = self.date_bucket_exposure.get(tags.date_bucket, 0.0) + stake
        self.source_exposure[tags.resolution_source] = self.source_exposure.get(tags.resolution_source, 0.0) + stake

    def release(self, tags: ClusterTags, stake: float) -> None:
        """Call when a position closes (resolution or early exit) to free
        up cluster capacity."""
        self.entity_exposure[tags.entity] = max(0.0, self.entity_exposure.get(tags.entity, 0.0) - stake)
        for theme in tags.themes:
            self.theme_exposure[theme] = max(0.0, self.theme_exposure.get(theme, 0.0) - stake)
        if tags.date_bucket is not None:
            self.date_bucket_exposure[tags.date_bucket] = max(
                0.0, self.date_bucket_exposure.get(tags.date_bucket, 0.0) - stake
            )
        self.source_exposure[tags.resolution_source] = max(
            0.0, self.source_exposure.get(tags.resolution_source, 0.0) - stake
        )


def ensure_event_tags(market: Market, gamma, tag_cache: dict[str, list[str]]) -> None:
    """Backfill `market.tags` from the parent event, memoised per event_id.

    Scans that paginate `/markets` (per-market volume order) get no event tags
    attached, but ordering by *event* volume instead is not an option: it
    front-loads the scan on a handful of mega multi-outcome events and, for
    Book M, collapses the overlap with the reward-eligible set from ~24% to
    ~3%. So we take the per-market ordering and pay for tags lazily, only for
    the handful of markets that reach the point of needing them.
    """
    if market.tags or not market.event_id:
        return
    if market.event_id not in tag_cache:
        try:
            tag_cache[market.event_id] = gamma.get_event(market.event_id).tags
        except Exception as exc:  # pragma: no cover - network failure path
            logger.warning("failed to fetch event tags for event=%s: %s", market.event_id, exc)
            tag_cache[market.event_id] = []
    market.tags = tag_cache[market.event_id]
