from __future__ import annotations

import pytest

from poly03.cluster.tagging import ClusterTags
from poly03.paper.state import PaperState, load_state, save_state


def _tags(market_id: str = "m1") -> ClusterTags:
    return ClusterTags(
        market_id=market_id,
        entity="Some Entity",
        themes=("politics",),
        geography="United States",
        resolution_source="example.gov",
        date_bucket="2026-09-01",
    )


def test_new_position_deducts_cash_and_computes_shares():
    state = PaperState(bankroll=10_000.0, cash=10_000.0, high_water_mark=10_000.0)
    pos = state.new_position(
        market_id="m1",
        question="Will X happen?",
        token_id="tok1",
        outcome="Yes",
        side_index=0,
        tier=1,
        entry_price=0.90,
        stake_usd=90.0,
        end_date=None,
        days_to_resolution_at_entry=30.0,
        modeled_annualized_roc=0.5,
        cluster_tags=_tags(),
    )
    assert state.cash == 10_000.0 - 90.0
    assert pos.shares == 90.0 / 0.90
    assert pos.status == "open"
    assert len(state.open_positions) == 1


def test_close_position_win_updates_cash_and_hwm():
    state = PaperState(bankroll=1_000.0, cash=1_000.0, high_water_mark=1_000.0)
    pos = state.new_position(
        market_id="m1",
        question="Will X happen?",
        token_id="tok1",
        outcome="Yes",
        side_index=0,
        tier=1,
        entry_price=0.90,
        stake_usd=90.0,
        end_date=None,
        days_to_resolution_at_entry=30.0,
        modeled_annualized_roc=0.5,
        cluster_tags=_tags(),
    )
    state.close_position(pos, status="resolved_win", reason="resolution", close_price=1.0)
    assert pos.status == "resolved_win"
    assert pos.realized_pnl == pytest.approx(90.0 / 0.90 - 90.0)
    assert state.cash == pytest.approx(1_000.0 - 90.0 + pos.stake_usd + pos.realized_pnl)
    assert state.high_water_mark >= 1_000.0


def test_close_position_loss_reduces_equity():
    state = PaperState(bankroll=1_000.0, cash=1_000.0, high_water_mark=1_000.0)
    pos = state.new_position(
        market_id="m1",
        question="Will X happen?",
        token_id="tok1",
        outcome="Yes",
        side_index=0,
        tier=1,
        entry_price=0.90,
        stake_usd=90.0,
        end_date=None,
        days_to_resolution_at_entry=30.0,
        modeled_annualized_roc=0.5,
        cluster_tags=_tags(),
    )
    state.close_position(pos, status="resolved_loss", reason="resolution", close_price=0.0)
    assert pos.realized_pnl == -90.0
    assert state.equity == pytest.approx(1_000.0 - 90.0)


def test_save_and_load_round_trip(tmp_path):
    state = PaperState(bankroll=5_000.0, cash=5_000.0, high_water_mark=5_000.0)
    state.new_position(
        market_id="m1",
        question="Will X happen?",
        token_id="tok1",
        outcome="Yes",
        side_index=0,
        tier=2,
        entry_price=0.88,
        stake_usd=50.0,
        end_date="2026-12-01T00:00:00+00:00",
        days_to_resolution_at_entry=60.0,
        modeled_annualized_roc=0.3,
        cluster_tags=_tags(),
    )
    path = tmp_path / "state.json"
    save_state(state, path)

    loaded = load_state(path)
    assert loaded.bankroll == 5_000.0
    assert len(loaded.positions) == 1
    p = loaded.positions[0]
    assert p.market_id == "m1"
    assert p.tier == 2
    assert p.cluster.entity == "Some Entity"


def test_load_state_missing_file_returns_defaults(tmp_path):
    state = load_state(tmp_path / "does_not_exist.json")
    assert state.positions == []
    assert state.cash == state.bankroll
