from __future__ import annotations

import pytest

from poly03.making.quoting import (
    build_quote_pair,
    inventory_skew,
    needs_requote,
    round_to_tick,
)
from poly03.making.rewards import RewardConfig


def _cfg(min_size=20.0, max_spread=4.5, daily=35.0):
    return RewardConfig(min_size=min_size, max_spread_cents=max_spread, daily_rate_usd=daily)


def _pair(**kw):
    params = dict(
        market_id="m1",
        question="q",
        token_id="t1",
        best_bid=0.48,
        best_ask=0.52,
        tick_size=0.01,
        reward=_cfg(),
        target_size_shares=100.0,
        net_inventory_shares=0.0,
        inventory_cap_shares=1_000.0,
    )
    params.update(kw)
    return build_quote_pair(**params)


def test_round_to_tick_modes():
    assert round_to_tick(0.4449, 0.01, mode="down") == 0.44
    assert round_to_tick(0.4451, 0.01, mode="up") == 0.45
    assert round_to_tick(0.4449, 0.01, mode="nearest") == 0.44


# --- quote placement --------------------------------------------------------


def test_improves_the_book_by_one_tick_when_there_is_room():
    pair = _pair(best_bid=0.48, best_ask=0.52)
    assert pair.bid.price == pytest.approx(0.49)
    assert pair.ask.price == pytest.approx(0.51)


def test_never_self_crosses_on_a_two_tick_spread():
    """At a two-tick spread, improving both sides would put bid == ask, so we
    join the touch instead."""
    pair = _pair(best_bid=0.49, best_ask=0.51)
    assert pair.bid is not None and pair.ask is not None
    assert pair.bid.price < pair.ask.price


def test_pulls_inside_the_reward_window_when_the_book_is_wider():
    """A 20c-wide book with a 4.5c scoring window: quoting one tick inside the
    touch would score zero, so we quote at the eligibility boundary instead."""
    pair = _pair(best_bid=0.40, best_ask=0.60, reward=_cfg(max_spread=4.5))
    mid = 0.50
    # safety margin is 0.5c, so the boundary is 4c from the midpoint
    assert pair.bid.distance_cents(mid) == pytest.approx(4.0)
    assert pair.ask.distance_cents(mid) == pytest.approx(4.0)
    assert pair.bid.price > 0.40  # pulled in from the touch
    assert pair.ask.price < 0.60


def test_emitted_quotes_always_score():
    pair = _pair(best_bid=0.40, best_ask=0.60)
    assert pair.qscore() > 0


# --- min_size cliff ---------------------------------------------------------


def test_size_below_reward_minimum_is_suppressed_not_shrunk():
    pair = _pair(target_size_shares=10.0, reward=_cfg(min_size=20.0))
    assert pair.is_empty
    assert "bid_below_reward_min_size" in pair.suppressed
    assert "ask_below_reward_min_size" in pair.suppressed


# --- inventory skew ---------------------------------------------------------


def test_inventory_skew_saturates():
    assert inventory_skew(500.0, 1_000.0) == pytest.approx(0.5)
    assert inventory_skew(5_000.0, 1_000.0) == 1.0
    assert inventory_skew(-5_000.0, 1_000.0) == -1.0
    assert inventory_skew(0.0, 0.0) == 0.0


def test_long_inventory_suppresses_the_bid_and_keeps_the_offer():
    pair = _pair(net_inventory_shares=1_000.0, inventory_cap_shares=1_000.0)
    assert pair.bid is None
    assert pair.ask is not None
    assert "bid_below_reward_min_size" in pair.suppressed


def test_short_inventory_suppresses_the_offer_and_keeps_the_bid():
    pair = _pair(net_inventory_shares=-1_000.0, inventory_cap_shares=1_000.0)
    assert pair.ask is None
    assert pair.bid is not None


def test_partial_inventory_shrinks_only_the_adding_side():
    pair = _pair(net_inventory_shares=500.0, inventory_cap_shares=1_000.0, target_size_shares=100.0)
    assert pair.bid.size_shares == pytest.approx(50.0)
    assert pair.ask.size_shares == pytest.approx(100.0)


# --- collateral -------------------------------------------------------------


def test_collateral_counts_both_sides():
    """Resting an ask on YES is economically buying NO, so it ties up capital
    too -- the caps are applied to the sum."""
    pair = _pair(best_bid=0.48, best_ask=0.52, target_size_shares=100.0)
    expected = 100.0 * pair.bid.price + 100.0 * (1.0 - pair.ask.price)
    assert pair.collateral_usd == pytest.approx(expected)


# --- requote ----------------------------------------------------------------


def test_needs_requote_on_a_material_mid_move():
    assert needs_requote(0.50, 0.52, threshold_cents=1.0)
    assert not needs_requote(0.50, 0.5005, threshold_cents=1.0)
