from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from poly03.cluster.tagging import ClusterTags
from poly03.paper import measurement as m
from poly03.paper.state import PaperState


def _tags(entity="E"):
    return ClusterTags(market_id="mx", entity=entity, themes=(), geography=None, resolution_source="src", date_bucket=None)


def _add_resolved(state, *, tier, entry_price, won, days_held=10.0, closed_at=None):
    pos = state.new_position(
        market_id=f"m-{len(state.positions)}",
        question="q",
        token_id="t",
        outcome="Yes",
        side_index=0,
        tier=tier,
        entry_price=entry_price,
        stake_usd=100.0,
        end_date=None,
        days_to_resolution_at_entry=days_held,
        modeled_annualized_roc=0.2,
        cluster_tags=_tags(),
    )
    close_price = 1.0 if won else 0.0
    state.close_position(pos, status="resolved_win" if won else "resolved_loss", reason="resolution", close_price=close_price)
    if closed_at is not None:
        pos.closed_at = closed_at.isoformat()
    else:
        opened = datetime.fromisoformat(pos.opened_at)
        pos.closed_at = (opened + timedelta(days=days_held)).isoformat()
    return pos


def test_calibration_by_tier_perfect_calibration():
    state = PaperState(bankroll=10_000.0, cash=10_000.0, high_water_mark=10_000.0)
    for _ in range(9):
        _add_resolved(state, tier=1, entry_price=0.90, won=True)
    _add_resolved(state, tier=1, entry_price=0.90, won=False)

    buckets = m.calibration_by_tier(state)
    assert len(buckets) == 1
    b = buckets[0]
    assert b.n == 10
    assert b.realized_win_rate == 0.9
    assert abs(b.mean_entry_price - 0.90) < 1e-9
    assert b.brier_score == pytest.approx((0.10**2 * 9 + 0.90**2 * 1) / 10)


def test_overall_brier_none_when_no_resolutions():
    state = PaperState(bankroll=10_000.0, cash=10_000.0, high_water_mark=10_000.0)
    assert m.overall_brier(state) is None


def test_early_exits_excluded_from_calibration():
    state = PaperState(bankroll=10_000.0, cash=10_000.0, high_water_mark=10_000.0)
    pos = state.new_position(
        market_id="m1",
        question="q",
        token_id="t",
        outcome="Yes",
        side_index=0,
        tier=1,
        entry_price=0.90,
        stake_usd=100.0,
        end_date=None,
        days_to_resolution_at_entry=10.0,
        modeled_annualized_roc=0.2,
        cluster_tags=_tags(),
    )
    state.close_position(pos, status="exited_early", reason="tier_downgrade", close_price=0.80)
    assert m.calibration_by_tier(state) == []
    assert m.overall_brier(state) is None


def test_pnl_by_tier_and_exit_reason_counts():
    state = PaperState(bankroll=10_000.0, cash=10_000.0, high_water_mark=10_000.0)
    _add_resolved(state, tier=1, entry_price=0.90, won=True)
    _add_resolved(state, tier=2, entry_price=0.88, won=False)

    pnl = m.pnl_by_tier(state)
    assert pnl[1] > 0
    assert pnl[2] < 0

    reasons = m.exit_reason_counts(state)
    assert reasons["resolution"] == 2


def test_phase1_gate_not_ready_below_min_resolutions():
    state = PaperState(bankroll=10_000.0, cash=10_000.0, high_water_mark=10_000.0)
    _add_resolved(state, tier=1, entry_price=0.90, won=True)
    status = m.phase1_gate_status(state)
    assert status.ready_to_advance is False
    assert status.n_resolutions == 1


def test_phase1_gate_blocked_by_tier1_miss():
    state = PaperState(bankroll=10_000.0, cash=10_000.0, high_water_mark=10_000.0)
    for _ in range(60):
        _add_resolved(state, tier=1, entry_price=0.95, won=True)
    _add_resolved(state, tier=1, entry_price=0.95, won=False)

    status = m.phase1_gate_status(state)
    assert status.tier1_misses == 1
    assert status.ready_to_advance is False


def test_phase1_gate_ready_when_all_conditions_met():
    state = PaperState(bankroll=10_000.0, cash=10_000.0, high_water_mark=10_000.0)
    # tier 3, not tier 1 -- a Tier 1 loss trips the gate on its own regardless
    # of calibration, so use a tier where a loss is an expected outcome.
    for i in range(55):
        _add_resolved(state, tier=3, entry_price=0.90, won=(i % 10 != 0))  # 90% win rate matches 0.90 price

    status = m.phase1_gate_status(state)
    assert status.n_resolutions == 55
    assert status.tier1_misses == 0
    assert status.calibration_ok is True
    assert status.ready_to_advance is True


def test_drawdown_stats_tracks_peak_to_trough():
    state = PaperState(bankroll=1_000.0, cash=1_000.0, high_water_mark=1_000.0)
    base = datetime.now(timezone.utc)
    _add_resolved(state, tier=1, entry_price=0.90, won=True, closed_at=base)
    _add_resolved(state, tier=1, entry_price=0.90, won=False, closed_at=base + timedelta(days=1))
    _add_resolved(state, tier=1, entry_price=0.90, won=False, closed_at=base + timedelta(days=2))

    dd = m.drawdown_stats(state)
    assert dd.max_drawdown_usd > 0
    assert 0.0 < dd.max_drawdown_fraction < 1.0
