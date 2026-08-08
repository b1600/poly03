"""§2.2 hard exclusion filters. Applied first, no exceptions -- a market
that trips any of these never reaches the classifier or scorer.

Each check is independently callable so the backtest / counterfactual log
(§7) can record *why* a market was rejected, not just that it was.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

from poly03.config import (
    AMBIGUOUS_BOILERPLATE_PATTERNS,
    AMBIGUOUS_RESOLUTION_KEYWORDS,
    BOOK_A_HORIZON_CAP_DAYS,
    MAX_SPREAD_BOOK_A,
    MIN_24H_VOLUME_USD,
    MIN_OPEN_INTEREST_USD,
)
from poly03.data.models import Market, OrderBook

# Phrases indicating the market can settle before its stated end_date / on a
# partial condition, which breaks the hold-to-expiry arithmetic in §1.1.
EARLY_RESOLUTION_KEYWORDS = (
    "will resolve early",
    "may resolve early",
    "resolve immediately if",
    "resolves immediately if",
    "resolve early if",
    "as soon as the outcome is known",
    "prior to the end date if",
)

# Coarse allowlist for "this description names a checkable, named source" --
# conservative by construction: anything not matching one of these patterns
# is treated as unreliable-source per §3.1's "unclassifiable -> no trade."
_NAMED_SOURCE_PATTERNS = (
    re.compile(r"https?://\S+"),
    re.compile(r"\bofficial\b", re.I),
    re.compile(r"\bgovernment\b", re.I),
    re.compile(r"\bcentral bank\b", re.I),
    re.compile(r"\belectoral commission\b", re.I),
    re.compile(r"\bsupreme court\b", re.I),
    re.compile(r"\bparliament\b", re.I),
    re.compile(r"\bcongress\b", re.I),
    re.compile(r"\bexchange\b", re.I),
    re.compile(r"\bleague\b", re.I),
    re.compile(r"\bfederal reserve\b", re.I),
)


class ExclusionReason(str, Enum):
    AMBIGUOUS_RESOLUTION = "ambiguous_resolution_criteria"
    UNRELIABLE_SOURCE = "unreliable_or_single_point_resolution_source"
    EARLY_PARTIAL_RESOLUTION = "early_partial_resolution_possible"
    THIN_BOOK = "thin_book"
    WIDE_SPREAD = "wide_spread"
    HORIZON_CAP = "resolution_date_beyond_horizon_cap"
    RESOLVING_OR_DISPUTE = "already_resolving_or_in_dispute"
    CLUSTER_FULL = "correlated_cluster_full"


@dataclass
class ExclusionResult:
    market_id: str
    reasons: list[ExclusionReason] = field(default_factory=list)
    ambiguous_keyword_hits: list[str] = field(default_factory=list)

    @property
    def excluded(self) -> bool:
        return len(self.reasons) > 0


_BOILERPLATE_RE = re.compile("|".join(AMBIGUOUS_BOILERPLATE_PATTERNS), re.I)


def _resolution_text(market: Market) -> str:
    return f"{market.description} {market.resolution_source}".lower()


def strip_resolution_boilerplate(text: str) -> str:
    """Remove Polymarket's stock UMA resolution language before ambiguity
    matching -- strategy_v2.md §1.2.

    "This market will resolve based on ... a consensus of credible reporting"
    appears on ~70% of all markets and says nothing about whether *this*
    market's criteria are objective. Matching on it rejected 60 of the 85
    in-band markets on grounds that applied to almost the entire venue.
    """
    return _BOILERPLATE_RE.sub(" ", text)


def check_ambiguous_resolution(market: Market) -> list[str]:
    text = strip_resolution_boilerplate(_resolution_text(market))
    return [kw for kw in AMBIGUOUS_RESOLUTION_KEYWORDS if kw in text]


def check_unreliable_source(market: Market) -> bool:
    """True if the market lacks a checkable, named resolution source."""
    if market.resolution_source.strip():
        return False
    text = f"{market.description}"
    return not any(p.search(text) for p in _NAMED_SOURCE_PATTERNS)


def check_early_partial_resolution(market: Market) -> bool:
    text = _resolution_text(market)
    return any(kw in text for kw in EARLY_RESOLUTION_KEYWORDS)


def check_thin_book(
    market: Market,
    *,
    min_oi_usd: float = MIN_OPEN_INTEREST_USD,
    min_24h_volume_usd: float = MIN_24H_VOLUME_USD,
) -> bool:
    """True (fails) if OI or 24h volume is below threshold, or unknown --
    unknown liquidity is treated as thin, not as a free pass."""
    if market.open_interest is None:
        return True
    if market.open_interest < min_oi_usd:
        return True
    return market.volume_24hr < min_24h_volume_usd


def check_wide_spread(
    market: Market,
    order_book: OrderBook | None = None,
    *,
    max_spread: float = MAX_SPREAD_BOOK_A,
) -> bool:
    """True (fails) if spread > max_spread, or unknown."""
    bid, ask = market.best_bid, market.best_ask
    if order_book is not None:
        bb, ba = order_book.best_bid, order_book.best_ask
        bid = bb.price if bb else bid
        ask = ba.price if ba else ask
    if bid is None or ask is None:
        return True
    return (ask - bid) > max_spread


def check_horizon_cap(market: Market, *, horizon_cap_days: float = BOOK_A_HORIZON_CAP_DAYS) -> bool:
    days = market.days_to_resolution
    if days is None:
        return True
    return days > horizon_cap_days


def check_resolving_or_dispute(market: Market) -> bool:
    if market.closed and not market.is_resolved:
        return True
    if not market.accepting_orders and not market.closed:
        return True
    unsettled_statuses = {"proposed", "disputed", "challenged"}
    if any(s in unsettled_statuses for s in market.uma_resolution_statuses):
        return True
    return False


def apply_resolution_risk_filters(market: Market) -> ExclusionResult:
    """The resolution-integrity subset of §2.2, for Book M (strategy_v2.md §3.1).

    Book M quotes rather than holds a directional view, so the liquidity and
    horizon gates in `apply_exclusion_filters` are wrong for it in both
    directions: `wide_spread` would reject exactly the markets that are most
    profitable to quote (a wide spread is the opportunity), and `horizon_cap`
    is a duration-risk control for a hold-to-resolution book that Book M is
    not. Book M applies its own liquidity/price/spread gates in
    making/universe.py.

    What does carry over is everything about *whether the market resolves
    cleanly*, because Book M will occasionally be caught holding inventory
    into a resolution. That is what this subset covers.
    """
    result = ExclusionResult(market_id=market.id)

    result.ambiguous_keyword_hits = check_ambiguous_resolution(market)
    if result.ambiguous_keyword_hits:
        result.reasons.append(ExclusionReason.AMBIGUOUS_RESOLUTION)

    if check_unreliable_source(market):
        result.reasons.append(ExclusionReason.UNRELIABLE_SOURCE)

    if check_early_partial_resolution(market):
        result.reasons.append(ExclusionReason.EARLY_PARTIAL_RESOLUTION)

    if check_resolving_or_dispute(market):
        result.reasons.append(ExclusionReason.RESOLVING_OR_DISPUTE)

    return result


def apply_exclusion_filters(
    market: Market,
    order_book: OrderBook | None = None,
    *,
    horizon_cap_days: float = BOOK_A_HORIZON_CAP_DAYS,
    min_oi_usd: float = MIN_OPEN_INTEREST_USD,
    min_24h_volume_usd: float = MIN_24H_VOLUME_USD,
    max_spread: float = MAX_SPREAD_BOOK_A,
    cluster_full: bool = False,
) -> ExclusionResult:
    result = ExclusionResult(market_id=market.id)

    result.ambiguous_keyword_hits = check_ambiguous_resolution(market)
    if result.ambiguous_keyword_hits:
        result.reasons.append(ExclusionReason.AMBIGUOUS_RESOLUTION)

    if check_unreliable_source(market):
        result.reasons.append(ExclusionReason.UNRELIABLE_SOURCE)

    if check_early_partial_resolution(market):
        result.reasons.append(ExclusionReason.EARLY_PARTIAL_RESOLUTION)

    if check_thin_book(market, min_oi_usd=min_oi_usd, min_24h_volume_usd=min_24h_volume_usd):
        result.reasons.append(ExclusionReason.THIN_BOOK)

    if check_wide_spread(market, order_book, max_spread=max_spread):
        result.reasons.append(ExclusionReason.WIDE_SPREAD)

    if check_horizon_cap(market, horizon_cap_days=horizon_cap_days):
        result.reasons.append(ExclusionReason.HORIZON_CAP)

    if check_resolving_or_dispute(market):
        result.reasons.append(ExclusionReason.RESOLVING_OR_DISPUTE)

    if cluster_full:
        result.reasons.append(ExclusionReason.CLUSTER_FULL)

    return result
