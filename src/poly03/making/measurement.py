"""§4 Phase 0 measurement for Book M.

The deliverable named in strategy_v2.md §4 is one number with an error bar:
our share of the reward pool. Everything here exists to produce it and to be
honest about how much it is worth.

Two things this report is careful *not* to claim:

1. The reward estimate rests on a reconstructed scoring formula
   (making/rewards.py). It is stated as an assumption on every report.
2. There is no P&L here. Spread capture and the maker rebate are real revenue
   lines (§3.3) but both depend on fills, and Phase 0 simulates none. A
   positive reward estimate is a necessary condition for Book M, not a
   sufficient one -- adverse selection (§3.4) can exceed it and is only
   observable with live orders.
"""

from __future__ import annotations

import statistics as st
from dataclasses import dataclass, field

from poly03.making.state import MakingState

# §4: Phase 0 runs a week at the scan cadence before its number means anything.
GATE_MIN_TICKS = 200
GATE_MIN_OBSERVATION_DAYS = 7.0
# Relative IQR above this means the estimate is too unstable to size against.
GATE_MAX_RELATIVE_DISPERSION = 0.75

SCORING_CAVEAT = (
    "ESTIMATE ONLY: reward figures come from a reconstruction of Polymarket's\n"
    "  published scoring formula (see making/rewards.py), not from realized\n"
    "  payouts. The reconstruction cannot be validated without resting real\n"
    "  orders -- that comparison is the first task of Phase 1 (§4). Do not\n"
    "  size real capital on these numbers alone."
)


@dataclass
class RewardEstimate:
    n_ticks: int
    observation_days: float
    median_usd_per_day: float
    p25_usd_per_day: float
    p75_usd_per_day: float
    median_collateral_usd: float
    median_pool_usd_per_day: float
    median_share_of_pool: float
    # The headline number: estimate excluding markets where no competing depth
    # was observed, so our share there is assumed rather than measured.
    median_identified_usd_per_day: float = 0.0
    median_unidentified_quoted: float = 0.0

    @property
    def relative_dispersion(self) -> float:
        if self.median_usd_per_day <= 0:
            return float("inf")
        return (self.p75_usd_per_day - self.p25_usd_per_day) / self.median_usd_per_day

    @property
    def annualized_yield_on_collateral(self) -> float:
        """Yield on the identified portion only. Using the total here would
        let uncontested pools -- where our share is assumed, not measured --
        drive the headline return."""
        if self.median_collateral_usd <= 0:
            return 0.0
        return self.median_identified_usd_per_day * 365.0 / self.median_collateral_usd


# Above this implied annualized yield, the estimate is telling us more about
# the model's blind spots than about the opportunity. Flagged, not suppressed.
IMPLAUSIBLE_ANNUAL_YIELD = 1.0


def implausibility_warning(est: "RewardEstimate") -> str | None:
    """A yield this high is not a forecast -- say so on the report.

    The estimator is a snapshot against today's book. It cannot see the two
    things that would actually cap the return: other makers arriving once a
    pool is visibly easy, and adverse selection on the fills that earn it.
    Both push the same way, and neither is observable before Phase 1.
    """
    if est.annualized_yield_on_collateral <= IMPLAUSIBLE_ANNUAL_YIELD:
        return None
    return (
        f"IMPLAUSIBLE YIELD ({est.annualized_yield_on_collateral:.0%}/yr): treat this as an\n"
        "  upper bound produced by a static snapshot, not a projection. It assumes the\n"
        "  competing book stays as thin as it is right now, which is precisely what will\n"
        "  stop being true if the opportunity is real. It also carries no cost for adverse\n"
        "  selection. The capacity of this strategy is bounded by both, and neither is\n"
        "  measurable until real orders rest (Phase 1)."
    )


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def reward_estimate(state: MakingState) -> RewardEstimate | None:
    ticks = [t for t in state.ticks if t.quoted > 0]
    if not ticks:
        return None
    daily = [t.total_est_reward_usd_per_day for t in ticks]
    collateral = [t.total_collateral_usd for t in ticks]
    pool = [t.pool_usd_per_day_in_quoted_markets for t in ticks]
    shares = [
        t.total_est_reward_usd_per_day / t.pool_usd_per_day_in_quoted_markets
        for t in ticks
        if t.pool_usd_per_day_in_quoted_markets > 0
    ]
    return RewardEstimate(
        n_ticks=len(state.ticks),
        observation_days=state.observation_days,
        median_usd_per_day=st.median(daily),
        p25_usd_per_day=_quantile(daily, 0.25),
        p75_usd_per_day=_quantile(daily, 0.75),
        median_collateral_usd=st.median(collateral),
        median_pool_usd_per_day=st.median(pool),
        median_share_of_pool=st.median(shares) if shares else 0.0,
        median_identified_usd_per_day=st.median([t.identified_est_reward_usd_per_day for t in ticks]),
        median_unidentified_quoted=st.median([t.unidentified_quoted for t in ticks]),
    )


@dataclass
class Phase0Gate:
    ready: bool
    blockers: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ready:
            return (
                "Phase 0 gate: READY -- the reward-share estimate is stable enough to act on.\n"
                "  Next: Phase 1 (§4), micro-live at ~$500, whose first job is to check the\n"
                "  reconstructed scoring and the realized maker fee against actual payouts."
            )
        return "Phase 0 gate: NOT READY\n" + "\n".join(f"  - {b}" for b in self.blockers)


def phase0_gate(state: MakingState) -> Phase0Gate:
    blockers: list[str] = []
    est = reward_estimate(state)

    if est is None:
        return Phase0Gate(ready=False, blockers=["no ticks with any quotable market yet"])
    if est.n_ticks < GATE_MIN_TICKS:
        blockers.append(f"{est.n_ticks} ticks recorded, need >={GATE_MIN_TICKS}")
    if est.observation_days < GATE_MIN_OBSERVATION_DAYS:
        blockers.append(f"{est.observation_days:.1f}d observed, need >={GATE_MIN_OBSERVATION_DAYS:.0f}d")
    if est.relative_dispersion > GATE_MAX_RELATIVE_DISPERSION:
        blockers.append(
            f"estimate too unstable: IQR/median = {est.relative_dispersion:.2f} "
            f"> {GATE_MAX_RELATIVE_DISPERSION}"
        )
    if est.median_identified_usd_per_day <= 0:
        blockers.append(
            "median *identified* reward estimate is zero -- either quotes are not "
            "scoring, or every pool we'd quote is uncontested and therefore unmeasured"
        )

    return Phase0Gate(ready=not blockers, blockers=blockers)


def universe_funnel(state: MakingState) -> list[tuple[str, int]]:
    """Aggregate rejection reasons across ticks, most common first. This is
    the Book M analogue of §7's counterfactual log: it says why the quotable
    universe is the size it is."""
    totals: dict[str, int] = {}
    for tick in state.ticks:
        for reason, n in tick.rejections.items():
            totals[reason] = totals.get(reason, 0) + n
    return sorted(totals.items(), key=lambda kv: -kv[1])


def top_markets_by_accrual(state: MakingState, limit: int = 15) -> list[tuple[str, float]]:
    ranked = sorted(state.market_reward_accrual.items(), key=lambda kv: -kv[1])
    return ranked[:limit]
