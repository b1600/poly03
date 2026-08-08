"""Polymarket liquidity-reward config + scoring reconstruction (strategy_v2.md §2.2, §4).

READ THIS BEFORE TRUSTING ANY NUMBER OUT OF THIS MODULE.

The reward *config* here is authoritative: `min_size`, `max_spread`, and
`rewards_daily_rate` come straight off the CLOB's /sampling-markets endpoint.

The reward *scoring* is a reconstruction of Polymarket's published formula.
Nobody outside the venue can verify it without resting real orders and
comparing the payout, which is exactly why strategy_v2.md §4 puts that
comparison in Phase 1 and not Phase 0. Everything downstream that consumes
`estimate_share()` is required to carry the caveat through to the operator --
`making/measurement.py` prints it on every report.

The formula, as implemented:

    for each resting order within `max_spread` cents of the midpoint,
        s     = |order_price - midpoint| in cents
        score = ((max_spread - s) / max_spread) ** exponent * size

    Qbid = sum of bid scores, Qask = sum of ask scores

    at the tails (midpoint <= 0.10 or >= 0.90) a one-sided quote qualifies:
        Q = Qbid + Qask
    otherwise two-sided quoting is required, and lopsided quoting is capped:
        Q = 0                 if either side is empty
        Q = 2 * Qmin          if Qmax / Qmin >= cutoff
        Q = Qmin + Qmax       otherwise

Our share of a market's daily rate is then Q_ours / (Q_ours + Q_others).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from poly03.config import (
    REWARD_MAX_ASSUMED_SHARE,
    REWARD_ONE_SIDED_PRICE_CEILING,
    REWARD_ONE_SIDED_PRICE_FLOOR,
    REWARD_ONE_SIDED_RATIO_CUTOFF,
    REWARD_SCORING_EXPONENT,
)
from poly03.data.models import OrderBook


@dataclass(frozen=True)
class RewardConfig:
    """Authoritative reward parameters for one market."""

    min_size: float           # minimum shares per order to score at all
    max_spread_cents: float   # scoring cutoff, in cents from the midpoint
    daily_rate_usd: float     # total pool paid across all makers, per day

    @property
    def funded(self) -> bool:
        return self.daily_rate_usd > 0 and self.max_spread_cents > 0 and self.min_size > 0

    @classmethod
    def from_clob_market(cls, raw: dict[str, Any]) -> RewardConfig | None:
        """Parse the `rewards` block of a /sampling-markets entry.

        Returns None when the market carries no usable reward config -- a
        surprising number of entries on that endpoint have `{"min_size": 0,
        "max_spread": 0}` with no rates, i.e. they are listed but not funded.
        """
        rewards = raw.get("rewards") or {}
        rates = rewards.get("rates") or []
        daily = 0.0
        for rate in rates:
            try:
                daily += float(rate.get("rewards_daily_rate") or 0.0)
            except (TypeError, ValueError):
                continue
        try:
            min_size = float(rewards.get("min_size") or 0.0)
            max_spread = float(rewards.get("max_spread") or 0.0)
        except (TypeError, ValueError):
            return None
        cfg = cls(min_size=min_size, max_spread_cents=max_spread, daily_rate_usd=daily)
        return cfg if cfg.funded else None


def order_score(
    price: float,
    size: float,
    midpoint: float,
    cfg: RewardConfig,
    *,
    exponent: float = REWARD_SCORING_EXPONENT,
) -> float:
    """Score one resting order. Zero if it is outside `max_spread` of the
    midpoint or below `min_size` -- both are cliffs, not gradients, which is
    why §3.2 treats `min_size` as a hard floor on our own order size."""
    if size < cfg.min_size or cfg.max_spread_cents <= 0:
        return 0.0
    distance_cents = abs(price - midpoint) * 100.0
    if distance_cents > cfg.max_spread_cents:
        return 0.0
    return ((cfg.max_spread_cents - distance_cents) / cfg.max_spread_cents) ** exponent * size


def combine_sides(qbid: float, qask: float, midpoint: float) -> float:
    """Apply the two-sided requirement to a pair of side scores."""
    if midpoint <= REWARD_ONE_SIDED_PRICE_FLOOR or midpoint >= REWARD_ONE_SIDED_PRICE_CEILING:
        return qbid + qask
    qmin, qmax = min(qbid, qask), max(qbid, qask)
    if qmin <= 0:
        return 0.0
    if qmax / qmin >= REWARD_ONE_SIDED_RATIO_CUTOFF:
        return 2.0 * qmin
    return qmin + qmax


def score_orders(orders: list[tuple[float, float]], midpoint: float, cfg: RewardConfig) -> float:
    """Sum `order_score` over (price, size) pairs on one side."""
    return sum(order_score(p, s, midpoint, cfg) for p, s in orders)


def _level_score(price: float, size: float, midpoint: float, cfg: RewardConfig) -> float:
    """Score one aggregated book level. Same decay as `order_score` but
    without the `min_size` floor -- see `book_qscore` for why."""
    if cfg.max_spread_cents <= 0 or size <= 0:
        return 0.0
    distance_cents = abs(price - midpoint) * 100.0
    if distance_cents > cfg.max_spread_cents:
        return 0.0
    return ((cfg.max_spread_cents - distance_cents) / cfg.max_spread_cents) ** REWARD_SCORING_EXPONENT * size


def book_qscore(book: OrderBook, midpoint: float, cfg: RewardConfig) -> float:
    """Total competing Q resting in the book -- the denominator of our share.

    Two deliberate differences from how we score *our own* quote, both because
    the public book is aggregated by price level and cannot be decomposed into
    individual makers:

    1. **No per-maker two-sided rule.** `combine_sides` encodes a penalty that
       Polymarket applies to each maker individually (one-sided scores zero;
       lopsided is capped at 2x the smaller side). Applying it to the pooled
       book treats every competitor as if they were one lopsided maker, which
       collapses the denominator in exactly the markets where competition is
       heaviest. Summing both sides is the correct aggregate.
    2. **No `min_size` floor.** The floor applies per order; a level holding
       three 100-share orders is three qualifying makers, not one sub-minimum
       one. Zeroing such levels silently deleted most of our competition.

    Both bugs pushed the same way -- they shrank the denominator and inflated
    our estimated share into four-figure annualized yields. The corrected
    version is biased the *other* way (it counts depth that may not qualify),
    which is the right direction to be wrong in for a number we would size
    capital against.
    """
    qbid = sum(_level_score(l.price, l.size, midpoint, cfg) for l in book.bids)
    qask = sum(_level_score(l.price, l.size, midpoint, cfg) for l in book.asks)
    return qbid + qask


@dataclass(frozen=True)
class ShareEstimate:
    share_fraction: float      # after the cap below
    usd_per_day: float
    raw_share_fraction: float  # before the cap, for diagnostics
    identified: bool           # was there any competing depth to measure against?
    capped: bool


def estimate_share(our_q: float, competing_q: float, cfg: RewardConfig) -> ShareEstimate:
    """Estimate our slice of one market's daily reward pool.

    Polymarket samples continuously and pays pro-rata by score, so a snapshot
    share is only as good as the assumption that the book keeps looking like
    this. Two guards on top of the raw pro-rata number:

    - **Identifiability.** With little or no competing depth inside the
      scoring window, the raw share approaches 100% for any size at all --
      the ratio is scale-free, so one minimum-size quote "wins" the whole
      pool. We require at least one qualifying competitor's worth of score
      (`min_size`) before treating the share as measured; below that we have
      no evidence about our competition, only its absence. Such markets are
      flagged `identified=False` so reports can quarantine them.

      This threshold matters more than it looks. A $126/day pool with 0.5 of
      competing score produced a $63/day estimate against a $19 quote -- a
      quarter of the whole book's estimate from one market. A pool that easy
      to capture would not stay uncontested; the model is missing the
      competition that would arrive, not finding free money.
    - **A hard ceiling** (`REWARD_MAX_ASSUMED_SHARE`) on what we will claim
      from any single pool, applied whether or not the market is identified.
    """
    identified = competing_q >= cfg.min_size

    total = our_q + competing_q
    if total <= 0 or our_q <= 0:
        return ShareEstimate(0.0, 0.0, 0.0, identified=identified, capped=False)

    raw = our_q / total
    share = min(raw, REWARD_MAX_ASSUMED_SHARE)
    return ShareEstimate(
        share_fraction=share,
        usd_per_day=share * cfg.daily_rate_usd,
        raw_share_fraction=raw,
        identified=identified,
        capped=share < raw,
    )
