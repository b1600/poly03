"""§3.1: which markets Book M is willing to quote.

The universe is the intersection of two sources:

- **CLOB /sampling-markets** -- authoritative for reward config and tick size,
  and by construction the only markets that pay rewards at all.
- **Gamma /markets** -- 24h volume, live best bid/ask, resolution text, and the
  event tags §4.3 cluster tagging needs.

They join on `condition_id`. Everything the venue tells us is in the CLOB feed;
everything about whether the market is *worth* quoting is in Gamma.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from poly03.classifier.llm_veto import LLMClassifierVeto, NoOpVeto
from poly03.classifier.pipeline import classify_market
from poly03.classifier.rules import Classification
from poly03.config import (
    GAMMA_MAX_SCAN_MARKETS,
    MAKING_FLATTEN_HOURS_BEFORE_RESOLUTION,
    MAKING_MAX_PRICE,
    MAKING_MIN_24H_VOLUME_USD,
    MAKING_MIN_PRICE,
    MAKING_MIN_REWARD_DAILY_RATE,
    MAKING_MIN_SPREAD_TICKS,
)
from poly03.data.clob import ClobClient
from poly03.data.gamma import GammaClient
from poly03.data.models import Market
from poly03.filters.exclusion import apply_resolution_risk_filters
from poly03.making.rewards import RewardConfig

logger = logging.getLogger("poly03.making")


@dataclass
class QuotableMarket:
    market: Market
    reward: RewardConfig
    classification: Classification
    yes_token_id: str
    no_token_id: str | None
    tick_size: float

    @property
    def midpoint(self) -> float:
        return (self.market.best_bid + self.market.best_ask) / 2.0

    @property
    def spread(self) -> float:
        return self.market.best_ask - self.market.best_bid

    @property
    def reward_density(self) -> float:
        """Daily reward rate per dollar of spread we have to cross to be
        competitive -- the ranking key when we can't quote everything."""
        return self.reward.daily_rate_usd


@dataclass
class UniverseReport:
    quotable: list[QuotableMarket] = field(default_factory=list)
    scanned: int = 0
    reward_eligible: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    def reject(self, reason: str) -> None:
        self.rejections[reason] = self.rejections.get(reason, 0) + 1


def load_reward_configs(clob: ClobClient, *, max_markets: int | None = None) -> dict[str, tuple[dict, RewardConfig]]:
    """condition_id -> (raw clob market, funded reward config)."""
    out: dict[str, tuple[dict, RewardConfig]] = {}
    for raw in clob.iter_sampling_markets(max_markets=max_markets):
        cfg = RewardConfig.from_clob_market(raw)
        if cfg is None:
            continue
        cond = raw.get("condition_id")
        if cond:
            out[cond] = (raw, cfg)
    return out


def select_universe(
    gamma: GammaClient,
    clob: ClobClient,
    *,
    veto: LLMClassifierVeto | None = None,
    max_gamma_markets: int = GAMMA_MAX_SCAN_MARKETS,
    max_sampling_markets: int | None = None,
    reward_configs: dict[str, tuple[dict, RewardConfig]] | None = None,
) -> UniverseReport:
    """Apply §3.1's gates and return the markets Book M would quote."""
    veto = veto or NoOpVeto()
    report = UniverseReport()

    configs = reward_configs if reward_configs is not None else load_reward_configs(clob, max_markets=max_sampling_markets)
    report.reward_eligible = len(configs)
    if not configs:
        logger.warning("no funded reward configs returned by /sampling-markets")
        return report

    # Sort key matters twice over.
    #
    # Per-market volume, not per-event: ordering by event volume front-loads
    # the scan on mega multi-outcome events whose legs are mostly unfunded,
    # dropping the overlap with the reward-eligible set from ~24% to ~3%.
    # Event tags are backfilled lazily in the engine for the few markets that
    # reach quoting -- see ensure_event_tags().
    #
    # And *24h* volume, not cumulative: cumulative volume is dominated by
    # markets that were busy months ago and are dead now, which is the exact
    # opposite of what §3.1 wants. Sorting by the same quantity we gate on
    # also lets us stop the scan the moment we cross the floor.
    for market in gamma.iter_markets(closed=False, order="volume24hr", ascending=False):
        if report.scanned >= max_gamma_markets:
            break
        report.scanned += 1

        if market.volume_24hr < MAKING_MIN_24H_VOLUME_USD:
            # Descending sort: everything after this point is quieter still.
            report.reject("below_min_24h_volume")
            break

        entry = configs.get(market.condition_id)
        if entry is None:
            report.reject("no_funded_rewards")
            continue
        raw_clob, reward = entry

        if reward.daily_rate_usd < MAKING_MIN_REWARD_DAILY_RATE:
            report.reject("reward_rate_below_floor")
            continue

        if not market.accepting_orders or market.closed:
            report.reject("not_accepting_orders")
            continue

        if market.best_bid is None or market.best_ask is None:
            report.reject("no_two_sided_quote")
            continue

        # §3.1: avoid the pinned tails, where inventory can't be exited and the
        # reward scoring degenerates to one-sided anyway.
        if not (MAKING_MIN_PRICE < market.best_bid and market.best_ask < MAKING_MAX_PRICE):
            report.reject("price_outside_quotable_band")
            continue

        tick = float(raw_clob.get("minimum_tick_size") or market.order_price_min_tick_size)
        if (market.best_ask - market.best_bid) < MAKING_MIN_SPREAD_TICKS * tick:
            report.reject("spread_too_tight_to_improve")
            continue

        days = market.days_to_resolution
        if days is None or days * 24.0 <= MAKING_FLATTEN_HOURS_BEFORE_RESOLUTION:
            report.reject("resolving_within_flatten_window")
            continue

        exclusion = apply_resolution_risk_filters(market)
        if exclusion.excluded:
            for reason in exclusion.reasons:
                report.reject(reason.value)
            continue

        # The tier is recorded but deliberately NOT used as a gate.
        #
        # strategy_v2.md §3.5 assumed the classifier could carry over as an
        # inventory-risk filter. It can't: Tier 4 means "requires a forecast",
        # which is a statement about *predictability*, not about resolution
        # integrity or inventory risk. A genuinely uncertain market is the
        # best thing a maker can quote -- two-way flow is where the spread and
        # the rewards are -- so excluding Tier 4 would throw away most of the
        # quotable universe for a reason that only applies to a directional
        # book. Resolution risk, the thing Book M actually cares about, is
        # handled by apply_resolution_risk_filters() above and by the
        # flatten-before-resolution window.
        #
        # It is kept on the QuotableMarket so reports can show the inventory
        # risk profile of what we're quoting.
        classification = classify_market(market, veto)

        token_ids = market.clob_token_ids
        if not token_ids:
            report.reject("no_clob_token_ids")
            continue

        report.quotable.append(
            QuotableMarket(
                market=market,
                reward=reward,
                classification=classification,
                yes_token_id=token_ids[0],
                no_token_id=token_ids[1] if len(token_ids) > 1 else None,
                tick_size=tick,
            )
        )

    report.quotable.sort(key=lambda q: q.reward_density, reverse=True)
    return report
