"""§8 Phase 0: reconstruct historical books and validate classifier tiers
against actual resolutions, with $0 capital and no live orders.

Data-availability caveat, stated once here rather than scattered in
comments: CLOB's /prices-history is sparse-to-empty for old, thin markets,
so the candidate set this engine can reconstruct is biased toward
markets that had enough volume to leave a price trail -- which is itself
a survivorship gap the doc calls out in §8 and §9.6. Treat this backtest
as a lower bound on sample size, and a first-order (not final) read on
calibration.

Exclusion filters are applied to each closed market's *current* (i.e.
final/closed) state, not its state at the historical entry point --
acceptable for a Phase 0 pass but a known approximation. It mainly
affects the RESOLVING_OR_DISPUTE and THIN_BOOK checks; ambiguous
resolution criteria / unreliable source are static text properties and
aren't affected either way.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from dataclasses import dataclass, field

from poly03.classifier.pipeline import classify_market
from poly03.classifier.taxonomy import Tier
from poly03.config import BOOK_A_HORIZON_CAP_DAYS, BOOK_A_PRICE_BAND
from poly03.data.clob import ClobClient
from poly03.data.gamma import GammaClient
from poly03.data.models import Market
from poly03.filters.exclusion import ExclusionReason, apply_exclusion_filters
from poly03.scoring.roc import annualized_roc, gross_return

logger = logging.getLogger("poly03.backtest")


@dataclass
class CandidateEntry:
    market_id: str
    question: str
    side: str
    side_index: int
    entry_price: float
    entry_timestamp: int
    days_to_resolution_at_entry: float
    tier: Tier
    confidence_multiplier: float
    exclusion_reasons: list[str]
    won: bool
    modeled_annualized_roc: float
    realized_gross_return: float
    realized_days_held: float

    @property
    def passed_filters(self) -> bool:
        return len(self.exclusion_reasons) == 0

    @property
    def tradeable(self) -> bool:
        lo, hi = BOOK_A_PRICE_BAND
        return self.passed_filters and self.tier != Tier.TIER_4 and lo <= self.entry_price <= hi


def _find_band_entry(
    history: list[tuple[int, float]],
    end_ts: float,
    *,
    band: tuple[float, float] = BOOK_A_PRICE_BAND,
    horizon_cap_days: float = BOOK_A_HORIZON_CAP_DAYS,
) -> tuple[int, float] | None:
    """Earliest point where price sits in-band with time-to-resolution
    still inside the horizon cap -- our proxy for 'this is when a maker
    order at the target price would first have made sense.'"""
    lo, hi = band
    for ts, price in history:
        days_remaining = (end_ts - ts) / 86400.0
        if days_remaining <= 0 or days_remaining > horizon_cap_days:
            continue
        if lo <= price <= hi:
            return ts, price
    return None


def build_candidates(market: Market, clob: ClobClient) -> list[CandidateEntry]:
    """One candidate per outcome side that ever traded into the Book A
    price band before resolution. Empty list if the market can't be used
    (not binary, unresolved, no end date, or no price history)."""
    if len(market.outcomes) != 2 or len(market.clob_token_ids) != 2:
        return []
    winning_idx = market.winning_outcome_index
    if winning_idx is None or market.end_date is None:
        return []

    end_ts = market.end_date.timestamp()
    classification = classify_market(market)
    # THIN_BOOK/WIDE_SPREAD reflect *current* liquidity/book state, which
    # collapses to ~0 volume and None bid/ask once a market closes -- we
    # have no historical depth data to check them against, so both are
    # disabled here (loosened thresholds + reason stripped below) rather
    # than trivially rejecting every candidate. RESOLVING_OR_DISPUTE is
    # also stripped: it partly keys off `umaResolutionStatuses`, which
    # Gamma appears to leave at its first-ever value (e.g. ["proposed"])
    # even after final resolution -- a live-monitoring signal that's stale
    # by the time a market is in the closed/historical set, and redundant
    # here anyway since build_candidates already gates on market.is_resolved.
    exclusion = apply_exclusion_filters(market, min_oi_usd=0.0, min_24h_volume_usd=0.0, max_spread=1.0)
    exclusion.reasons = [
        r for r in exclusion.reasons if r not in (ExclusionReason.WIDE_SPREAD, ExclusionReason.RESOLVING_OR_DISPUTE)
    ]

    out: list[CandidateEntry] = []
    for idx, (outcome, token_id) in enumerate(zip(market.outcomes, market.clob_token_ids)):
        try:
            history = clob.get_price_history(token_id)
        except Exception as exc:
            logger.warning("price history fetch failed market=%s token=%s: %s", market.id, token_id, exc)
            continue
        found = _find_band_entry(history, end_ts)
        if found is None:
            continue
        ts, price = found
        days_at_entry = (end_ts - ts) / 86400.0
        won = idx == winning_idx
        gross = gross_return(price) if won else -1.0
        try:
            modeled_roc = annualized_roc(price, days_at_entry)
        except ValueError:
            modeled_roc = 0.0

        out.append(
            CandidateEntry(
                market_id=market.id,
                question=market.question,
                side=outcome,
                side_index=idx,
                entry_price=price,
                entry_timestamp=ts,
                days_to_resolution_at_entry=days_at_entry,
                tier=classification.tier,
                confidence_multiplier=classification.confidence_multiplier,
                exclusion_reasons=[r.value for r in exclusion.reasons],
                won=won,
                modeled_annualized_roc=modeled_roc,
                realized_gross_return=gross,
                realized_days_held=days_at_entry,
            )
        )
    return out


@dataclass
class CalibrationBucket:
    tier: Tier
    n: int
    mean_entry_price: float  # avg "implied probability" we were pricing in
    realized_win_rate: float
    brier_score: float


@dataclass
class BacktestReport:
    candidates: list[CandidateEntry] = field(default_factory=list)

    @property
    def tradeable(self) -> list[CandidateEntry]:
        return [c for c in self.candidates if c.tradeable]

    def calibration_by_tier(self) -> list[CalibrationBucket]:
        buckets: dict[Tier, list[CandidateEntry]] = {}
        for c in self.tradeable:
            buckets.setdefault(c.tier, []).append(c)
        out = []
        for tier, items in sorted(buckets.items(), key=lambda kv: kv[0].value):
            n = len(items)
            mean_p = sum(i.entry_price for i in items) / n
            win_rate = sum(1 for i in items if i.won) / n
            brier = sum((i.entry_price - (1.0 if i.won else 0.0)) ** 2 for i in items) / n
            out.append(
                CalibrationBucket(tier=tier, n=n, mean_entry_price=mean_p, realized_win_rate=win_rate, brier_score=brier)
            )
        return out

    def overall_brier(self) -> float | None:
        items = self.tradeable
        if not items:
            return None
        return sum((i.entry_price - (1.0 if i.won else 0.0)) ** 2 for i in items) / len(items)

    def realized_vs_modeled_roc(self) -> tuple[float, float] | None:
        """(mean modeled annualized ROC, mean realized annualized ROC) over
        tradeable candidates. The gap is fees + slippage + lockup we
        underestimated once real trading starts (§7) -- though this
        backtest can't measure fees/slippage at all, so a nonzero gap here
        only reflects classifier/timing error, not execution cost."""
        items = self.tradeable
        if not items:
            return None
        modeled = sum(i.modeled_annualized_roc for i in items) / len(items)
        realized = []
        for i in items:
            if i.realized_days_held > 0:
                realized.append((1 + i.realized_gross_return) ** (365.0 / i.realized_days_held) - 1)
            else:
                realized.append(i.realized_gross_return)
        return modeled, sum(realized) / len(realized)

    def counterfactual_rejected(self) -> list[CandidateEntry]:
        """§7: what rejected markets would have returned -- tells us if the
        filters are too tight."""
        return [c for c in self.candidates if not c.passed_filters]

    def tier1_misses(self) -> list[CandidateEntry]:
        """§4.4 kill-switch condition: any Tier 1 position resolving against
        us means the classifier is broken, not unlucky."""
        return [c for c in self.tradeable if c.tier == Tier.TIER_1 and not c.won]


def run_phase0_backtest(
    *,
    max_markets: int = 100,
    gamma: GammaClient | None = None,
    clob: ClobClient | None = None,
    page_size: int = 20,
) -> BacktestReport:
    gamma = gamma or GammaClient()
    clob = clob or ClobClient()

    candidates: list[CandidateEntry] = []
    n = 0
    for market in gamma.iter_markets_with_event_context(
        closed=True, order="volume", ascending=False, page_size=page_size
    ):
        if n >= max_markets:
            break
        n += 1
        try:
            candidates.extend(build_candidates(market, clob))
        except Exception as exc:
            logger.warning("failed to build candidates for market=%s: %s", market.id, exc)

    return BacktestReport(candidates=candidates)


def iter_closed_markets_for_backtest(gamma: GammaClient, max_markets: int) -> Iterator[Market]:
    """Exposed separately so callers (e.g. the CLI) can report progress
    while the backtest runs, since price-history fetches are the slow part."""
    n = 0
    for market in gamma.iter_markets_with_event_context(closed=True, order="volume", ascending=False):
        if n >= max_markets:
            return
        n += 1
        yield market
