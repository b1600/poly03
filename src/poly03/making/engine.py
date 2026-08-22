"""Book M tick (strategy_v2.md §3 + §4 Phase 0).

One `run_tick()` = one scan cadence:

    select universe (§3.1)
      -> allocate capital across the top markets (§3.2, §3.4)
        -> build the two-sided quote we would rest
          -> score it against the live book and estimate our reward share (§4)

**No orders are placed and no fills are simulated.** That is not a stub: §4
scopes Phase 0 to the one thing that can be measured without capital -- what
share of the reward pool our quotes would score against observed competing
depth. Fill rate, the realized maker fee, and adverse selection are Phase 1.

Sizing note. We quote exactly `reward.min_size` per side by default rather
than the largest size the caps allow. Reward score is linear in size and so is
collateral, so score-per-dollar is roughly flat within a market; spreading the
same capital across more markets buys diversification of both inventory and
reward-program risk for free. Size up only when the pool in a single market is
large enough to be worth the concentration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from poly03.classifier.llm_veto import LLMClassifierVeto, NoOpVeto
from poly03.cluster.tagging import ClusterExposureTracker, ensure_event_tags, tag_market
from poly03.config import (
    GAMMA_MAX_SCAN_MARKETS,
    MAKING_DECISION_LOG_FILE,
    MAKING_MAX_DEPLOYED_FRACTION,
    MAKING_MAX_INVENTORY_PER_MARKET_FRACTION,
    MAKING_MAX_MARKETS_QUOTED,
)
from poly03.data.clob import ClobClient
from poly03.data.gamma import GammaClient
from poly03.making.quoting import build_quote_pair
from poly03.making.rewards import book_qscore, estimate_share
from poly03.making.state import MakingState, MakingTickSummary, MarketObservation, log_observation
from poly03.making.universe import QuotableMarket, UniverseReport, select_universe

logger = logging.getLogger("poly03.making")


@dataclass
class MakingTickReport:
    timestamp: str
    universe: UniverseReport
    observations: list[MarketObservation] = field(default_factory=list)
    summary: MakingTickSummary | None = None
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _inventory_cap_shares(bankroll: float, price: float, fraction: float = MAKING_MAX_INVENTORY_PER_MARKET_FRACTION) -> float:
    """§3.4 per-market inventory cap, converted from USD to shares.
    `fraction` defaults to Phase 0's constant; execution.py's live engine
    passes MAKING_LIVE_MAX_INVENTORY_PER_MARKET_FRACTION instead -- Phase 0's
    bankroll is orders of magnitude larger than Phase 1's real bankroll_cap,
    so the same fraction would reject every market's min_size at live scale
    (task 20260818_2012 item 1a)."""
    if price <= 0:
        return 0.0
    return (fraction * bankroll) / price


def run_tick(
    state: MakingState,
    *,
    gamma: GammaClient | None = None,
    clob: ClobClient | None = None,
    veto: LLMClassifierVeto | None = None,
    max_gamma_markets: int = GAMMA_MAX_SCAN_MARKETS,
    max_markets_quoted: int = MAKING_MAX_MARKETS_QUOTED,
    decision_log_path: str = MAKING_DECISION_LOG_FILE,
) -> MakingTickReport:
    gamma = gamma or GammaClient()
    clob = clob or ClobClient()
    veto = veto or NoOpVeto()

    universe = select_universe(gamma, clob, veto=veto, max_gamma_markets=max_gamma_markets)
    report = MakingTickReport(timestamp=_now_iso(), universe=universe)

    budget = MAKING_MAX_DEPLOYED_FRACTION * state.bankroll
    spent = 0.0
    cluster_tracker = ClusterExposureTracker(bankroll=state.bankroll)
    tag_cache: dict[str, list[str]] = {}

    selected: list[QuotableMarket] = universe.quotable[: max_markets_quoted * 3]
    books = _fetch_books(clob, selected, report)

    for qm in selected:
        if len(report.observations) >= max_markets_quoted:
            break

        book = books.get(qm.yes_token_id)
        if book is None:
            report.skip("no_order_book")
            continue

        bb, ba = book.best_bid, book.best_ask
        if bb is None or ba is None or ba.price <= bb.price:
            report.skip("book_not_two_sided")
            continue
        best_bid, best_ask = bb.price, ba.price
        midpoint = (best_bid + best_ask) / 2.0

        target_shares = qm.reward.min_size
        cap_shares = _inventory_cap_shares(state.bankroll, midpoint)
        if target_shares > cap_shares:
            # The venue's minimum scoring size is larger than our per-market
            # risk budget allows. Quoting anyway would breach §3.4; quoting
            # smaller would score zero. Neither is acceptable -- skip.
            report.skip("reward_min_size_exceeds_inventory_cap")
            continue

        pair = build_quote_pair(
            market_id=qm.market.id,
            question=qm.market.question,
            token_id=qm.yes_token_id,
            best_bid=best_bid,
            best_ask=best_ask,
            tick_size=qm.tick_size,
            reward=qm.reward,
            target_size_shares=target_shares,
            net_inventory_shares=0.0,  # Phase 0 simulates no fills, so flat
            inventory_cap_shares=cap_shares,
        )
        if pair.is_empty:
            report.skip("no_eligible_quote")
            continue

        if spent + pair.collateral_usd > budget:
            report.skip("deployment_budget_exhausted")
            continue

        ensure_event_tags(qm.market, gamma, tag_cache)
        tags = tag_market(qm.market)
        breaches = cluster_tracker.would_breach(tags, pair.collateral_usd)
        if breaches:
            report.skip("cluster_cap")
            continue

        our_q = pair.qscore()
        competing_q = book_qscore(book, midpoint, qm.reward)
        share = estimate_share(our_q, competing_q, qm.reward)

        cluster_tracker.register(tags, pair.collateral_usd)
        spent += pair.collateral_usd

        obs = MarketObservation(
            market_id=qm.market.id,
            condition_id=qm.market.condition_id,
            question=qm.market.question,
            midpoint=midpoint,
            best_bid=best_bid,
            best_ask=best_ask,
            tick_size=qm.tick_size,
            reward_daily_rate=qm.reward.daily_rate_usd,
            reward_min_size=qm.reward.min_size,
            reward_max_spread_cents=qm.reward.max_spread_cents,
            our_qscore=our_q,
            competing_qscore=competing_q,
            share_fraction=share.share_fraction,
            est_reward_usd_per_day=share.usd_per_day,
            identified=share.identified,
            raw_share_fraction=share.raw_share_fraction,
            collateral_usd=pair.collateral_usd,
            bid_price=pair.bid.price if pair.bid else None,
            bid_size=pair.bid.size_shares if pair.bid else None,
            ask_price=pair.ask.price if pair.ask else None,
            ask_size=pair.ask.size_shares if pair.ask else None,
            suppressed=list(pair.suppressed),
        )
        report.observations.append(obs)
        log_observation({"kind": "quote", **obs.__dict__}, path=decision_log_path)

    report.summary = MakingTickSummary(
        timestamp=report.timestamp,
        gamma_scanned=universe.scanned,
        reward_eligible=universe.reward_eligible,
        quotable=len(universe.quotable),
        quoted=len(report.observations),
        total_collateral_usd=spent,
        total_est_reward_usd_per_day=sum(o.est_reward_usd_per_day for o in report.observations),
        pool_usd_per_day_in_quoted_markets=sum(o.reward_daily_rate for o in report.observations),
        unidentified_est_reward_usd_per_day=sum(
            o.est_reward_usd_per_day for o in report.observations if not o.identified
        ),
        unidentified_quoted=sum(1 for o in report.observations if not o.identified),
        rejections=dict(universe.rejections),
    )
    state.record(report.summary, report.observations)
    return report


def _fetch_books(clob: ClobClient, markets: list[QuotableMarket], report: MakingTickReport) -> dict:
    """Batch the order-book reads. One request per ~100 tokens beats one per
    market at this universe size."""
    token_ids = [qm.yes_token_id for qm in markets]
    books: dict = {}
    for i in range(0, len(token_ids), 100):
        chunk = token_ids[i : i + 100]
        try:
            books.update(clob.get_order_books(chunk))
        except Exception as exc:
            logger.warning("order-book batch failed (%d tokens): %s", len(chunk), exc)
            report.skip("order_book_fetch_failed")
    return books
