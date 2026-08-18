"""strategy_v2.md §4 Phase 1 measurement: does Book M actually work with real
capital? Phase 0 (`making/measurement.py`) could only ever produce an
estimate under its own `SCORING_CAVEAT` -- no fills, no fees, no adverse
selection. This module answers the three things the §4 rollout plan names
as Phase 1's job:

1. Realized vs. estimated reward capture -- does `making/rewards.py`'s
   scoring reconstruction hold up against real, manually-logged payouts
   (`live_state.py`'s `record_reward_payout` -- see that module for why
   rewards can't be reconciled automatically)?
2. Fill rate on resting quotes -- meaningless in Phase 0, where every
   resting order is *assumed* filled (see paper/engine.py's docstring for
   the same point about Book A). Reconstructed here from the decision log,
   since a cancelled-unfilled order isn't kept in `state.open_orders` once
   it's gone.
3. The §4 gate itself: rewards + rebate + spread capture against realized
   adverse selection (`making/execution.py`'s markout series), read out
   against the plan's explicit >=500-fill threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from poly03.making.live_state import LiveMakingState
from poly03.making.state import MakingState

GATE_MIN_FILLS = 500
# Below this, dividing a real payout by an almost-zero window produces a
# meaningless (and alarmingly large) rate rather than an early estimate.
MIN_OBSERVATION_DAYS_FOR_RATE = 1.0 / 24.0  # 1 hour


@dataclass
class RewardComparison:
    realized_total_usd: float
    realized_usd_per_day: float | None
    n_payouts: int
    observation_days: float
    estimated_usd_per_day: float | None = None

    @property
    def ratio(self) -> float | None:
        """realized / estimated. <1 means the Phase 0 reconstruction
        (making/rewards.py) overstated what we'd actually capture; the
        Phase 0 report already warns its number is an upper bound, so a
        ratio well below 1 is expected, not necessarily a bug."""
        if not self.estimated_usd_per_day or self.realized_usd_per_day is None:
            return None
        return self.realized_usd_per_day / self.estimated_usd_per_day


def reward_capture_comparison(state: LiveMakingState, phase0_state: MakingState | None = None) -> RewardComparison:
    days = (datetime.now(timezone.utc) - datetime.fromisoformat(state.created_at)).total_seconds() / 86400.0
    realized_per_day = state.realized_reward_usd_total / days if days >= MIN_OBSERVATION_DAYS_FOR_RATE else None

    estimated_per_day = None
    if phase0_state is not None:
        from poly03.making.measurement import reward_estimate

        est = reward_estimate(phase0_state)
        if est is not None:
            estimated_per_day = est.median_identified_usd_per_day

    return RewardComparison(
        realized_total_usd=state.realized_reward_usd_total,
        realized_usd_per_day=realized_per_day,
        n_payouts=len(state.reward_payouts),
        observation_days=days,
        estimated_usd_per_day=estimated_per_day,
    )


@dataclass
class FillRateStats:
    n_orders_placed: int
    n_orders_filled: int
    notional_placed_usd: float
    notional_filled_usd: float

    @property
    def order_count_fill_rate(self) -> float | None:
        return self.n_orders_filled / self.n_orders_placed if self.n_orders_placed else None

    @property
    def notional_fill_rate(self) -> float | None:
        return self.notional_filled_usd / self.notional_placed_usd if self.notional_placed_usd else None


def fill_rate(decision_log_path: str | Path) -> FillRateStats:
    """A "filled" order is one that got at least one recorded fill --
    partial fills count. Orders that only ever got cancelled (stale
    requote, unwind) never appear here as filled, which is the point: this
    is the number that tells us whether resting quotes are actually
    reachable, not whether we tried to place them."""
    placed: dict[str, float] = {}
    filled_ids: set[str] = set()

    p = Path(decision_log_path)
    if not p.exists():
        return FillRateStats(0, 0, 0.0, 0.0)

    with p.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            kind = event.get("kind")
            order_id = event.get("order_id")
            if kind == "place" and order_id:
                placed[order_id] = float(event.get("collateral_usd") or 0.0)
            elif kind == "fill" and order_id:
                filled_ids.add(order_id)

    filled_ids &= placed.keys()
    return FillRateStats(
        n_orders_placed=len(placed),
        n_orders_filled=len(filled_ids),
        notional_placed_usd=sum(placed.values()),
        notional_filled_usd=sum(placed[i] for i in filled_ids),
    )


@dataclass
class AdverseSelectionSummary:
    n_fills: int
    n_scored: int
    spread_capture_usd: float  # sum of favorable markouts (positive)
    adverse_selection_usd: float  # sum of |unfavorable markouts| (positive)
    reward_usd: float
    fee_usd: float

    @property
    def capture_usd(self) -> float:
        """rewards + rebate + spread capture, per §4's gate wording. Fees
        are subtracted as a straight cost rather than netted against a
        separate rebate line -- making/execution.py's `_fee_usd` doesn't
        yet distinguish a maker rebate from a taker fee (§2.1's own
        caveat), so until that's confirmed against real fills this treats
        any recorded fee as pure cost, which understates capture if a
        rebate is actually landing."""
        return self.reward_usd - self.fee_usd + self.spread_capture_usd

    @property
    def net_usd(self) -> float:
        return self.capture_usd - self.adverse_selection_usd


def adverse_selection_summary(state: LiveMakingState) -> AdverseSelectionSummary:
    """Uses the 30-minute markout where it's matured, falling back to the
    5-minute one -- the longer horizon is the less noisy read on whether a
    fill was actually adverse, but requiring it for every fill would throw
    away the most recent half hour of fills on every report."""
    spread_capture = 0.0
    adverse = 0.0
    n_scored = 0
    for fill in state.fills:
        m = fill.markout_30m_usd if fill.markout_30m_usd is not None else fill.markout_5m_usd
        if m is None:
            continue
        n_scored += 1
        if m >= 0:
            spread_capture += m
        else:
            adverse += -m
    return AdverseSelectionSummary(
        n_fills=len(state.fills),
        n_scored=n_scored,
        spread_capture_usd=spread_capture,
        adverse_selection_usd=adverse,
        reward_usd=state.realized_reward_usd_total,
        fee_usd=state.realized_fee_usd_total,
    )


@dataclass
class Phase1Gate:
    ready: bool
    blockers: list[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.ready:
            return (
                f"Phase 1 gate: READY -- capture exceeds realized adverse selection over >={GATE_MIN_FILLS} fills.\n"
                "  Next: Phase 2 (§4), scale to the $10k book, capped by per-market depth share."
            )
        return "Phase 1 gate: NOT READY\n" + "\n".join(f"  - {b}" for b in self.blockers)


def phase1_gate(state: LiveMakingState) -> Phase1Gate:
    blockers: list[str] = []
    summary = adverse_selection_summary(state)

    if summary.n_fills < GATE_MIN_FILLS:
        blockers.append(f"{summary.n_fills} fills recorded, need >={GATE_MIN_FILLS}")
    if summary.n_scored == 0:
        blockers.append("no fills old enough yet to have a markout (needs 5m+)")
    elif summary.net_usd <= 0:
        blockers.append(
            f"capture (${summary.capture_usd:,.2f} = reward + spread capture - fees) does not exceed "
            f"realized adverse selection (${summary.adverse_selection_usd:,.2f})"
        )
    if state.halted:
        blockers.append(f"state is halted: {'; '.join(state.halt_reasons)}")

    return Phase1Gate(ready=not blockers, blockers=blockers)
