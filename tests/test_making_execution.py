from __future__ import annotations

from datetime import datetime, timedelta, timezone

from poly03.classifier.rules import Classification
from poly03.classifier.taxonomy import Tier
from poly03.data.models import OrderBook
from poly03.making.execution import (
    LiveTickReport,
    _place_side,
    _rank_affordable,
    check_adverse_selection_kill_switch,
    check_drawdown_kill_switch,
    compute_markouts,
    run_live_tick,
)
from poly03.making.live_state import LiveMakingState
from poly03.making.quoting import build_quote_pair
from poly03.making.rewards import RewardConfig
from poly03.making.universe import QuotableMarket, UniverseReport


def _qm(market, *, min_size=20, max_spread=4.5, daily=35.0, tick=0.01, no_token_id="222"):
    reward = RewardConfig(min_size=min_size, max_spread_cents=max_spread, daily_rate_usd=daily)
    classification = Classification(market_id=market.id, tier=Tier.TIER_4, confidence_multiplier=0.0)
    return QuotableMarket(
        market=market,
        reward=reward,
        classification=classification,
        yes_token_id="111",
        no_token_id=no_token_id,
        tick_size=tick,
    )


def _book(best_bid=0.48, best_ask=0.52, size=500.0):
    return OrderBook(asset_id="111", bids=[{"price": best_bid, "size": size}], asks=[{"price": best_ask, "size": size}])


# Matches test_making_engine.py's QUOTABLE_DESC/resolution_source -- without
# these, select_universe's resolution-risk filters reject the market before
# it's ever quotable, independent of anything this file is testing.
QUOTABLE_DESC = (
    "The winner will be determined by the official election commission's "
    "certified results. This market will resolve based on a consensus of "
    "credible reporting."
)


def _quotable_market(market_factory, **kw):
    params = dict(
        description=QUOTABLE_DESC,
        resolution_source="Official election commission",
        best_bid=0.48,
        best_ask=0.52,
        volume_24hr=50_000.0,
        days_to_resolution=90.0,
    )
    params.update(kw)
    return market_factory("Test market?", **params)


class FakeGamma:
    def __init__(self, markets=()):
        self._markets = list(markets)

    def iter_markets(self, **kwargs):
        yield from self._markets

    def get_event(self, event_id):
        raise AssertionError("get_event should not be called when market.event_id is unset in tests")


class FakeClob:
    def __init__(self, sampling=(), books=None):
        self._sampling = list(sampling)
        self._books = books or {}
        self.posted: list[dict] = []
        self.cancelled: list[list[str]] = []
        self._post_side_effect = None

    def set_post_side_effect(self, fn):
        self._post_side_effect = fn

    def iter_sampling_markets(self, *, max_markets=None):
        yield from self._sampling

    def get_order_books(self, token_ids):
        return {t: self._books[t] for t in token_ids if t in self._books}

    def get_order_book(self, token_id):
        return self._books[token_id]

    def get_fee_rate_bps(self, token_id):
        return None

    def get_open_orders(self, **kw):
        return []

    def post_limit_order(self, *, token_id, price, size, side, tick_size, neg_risk):
        if self._post_side_effect is not None:
            self._post_side_effect(token_id=token_id, price=price, size=size, side=side)
        self.posted.append(dict(token_id=token_id, price=price, size=size, side=side))
        return {"orderID": f"order-{len(self.posted)}"}

    def cancel_orders(self, ids):
        self.cancelled.append(list(ids))
        return {}


def _sampling_entry(condition_id="0xabc", min_size=20, max_spread=4.5, daily=35, tick=0.01):
    return {
        "condition_id": condition_id,
        "minimum_tick_size": tick,
        "rewards": {"rates": [{"rewards_daily_rate": daily}], "min_size": min_size, "max_spread": max_spread},
    }


# --- NO-leg routing (task item 2) -------------------------------------------


def test_ask_leg_routes_to_no_token_as_a_buy_not_a_sell(market_factory):
    """Polymarket has no naked shorting -- an ask must be a BUY on the NO
    token at 1-price, never a SELL on the YES token."""
    market = market_factory("Test?", best_bid=0.48, best_ask=0.52)
    qm = _qm(market)
    pair = build_quote_pair(
        market_id=market.id,
        question=market.question,
        token_id="111",
        best_bid=0.48,
        best_ask=0.52,
        tick_size=0.01,
        reward=qm.reward,
        target_size_shares=20,
        inventory_cap_shares=100,
    )
    state = LiveMakingState(bankroll_cap_usd=100, cash_usd=100)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())
    clob = FakeClob()

    result = _place_side(state, clob, qm, pair.ask, pair, report, dry_run=False, decision_log_path="/tmp/x.jsonl", cluster_tags={})

    assert result is not None
    assert clob.posted[0]["token_id"] == "222"  # NO token, not YES
    assert clob.posted[0]["side"] == "BUY"
    assert abs(clob.posted[0]["price"] - (1.0 - pair.ask.price)) < 1e-9


def test_ask_leg_fails_cleanly_without_a_no_token(market_factory):
    market = market_factory("Test?", best_bid=0.48, best_ask=0.52)
    qm = _qm(market, no_token_id=None)
    pair = build_quote_pair(
        market_id=market.id,
        question=market.question,
        token_id="111",
        best_bid=0.48,
        best_ask=0.52,
        tick_size=0.01,
        reward=qm.reward,
        target_size_shares=20,
        inventory_cap_shares=100,
    )
    state = LiveMakingState(bankroll_cap_usd=100, cash_usd=100)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())
    clob = FakeClob()

    result = _place_side(state, clob, qm, pair.ask, pair, report, dry_run=False, decision_log_path="/tmp/x.jsonl", cluster_tags={})

    assert result is None
    assert clob.posted == []
    assert any("no NO token id" in e for e in report.errors)


# --- per-token inventory netting (task item 2) ------------------------------


def test_yes_and_no_fills_net_independently_not_clobbered(market_factory):
    state = LiveMakingState(bankroll_cap_usd=100, cash_usd=100)
    state.record_fill(
        market_id="m1", condition_id="0xabc", token_id="111", question="q", side="buy", price=0.49, size_shares=20, order_id="o1"
    )
    state.record_fill(
        market_id="m1", condition_id="0xabc", token_id="222", question="q", side="buy", price=0.49, size_shares=20, order_id="o2"
    )
    assert len(state.positions) == 2
    yes_pos = state.position_for_token("m1", "111")
    no_pos = state.position_for_token("m1", "222")
    assert yes_pos.net_shares == 20
    assert no_pos.net_shares == 20  # not clobbered into one row


# --- paired placement rollback (task item 5) --------------------------------


def test_partial_leg_failure_rolls_back_the_successful_leg(market_factory):
    market = _quotable_market(market_factory)
    market.id = "test-1"

    def fail_no_leg(*, token_id, price, size, side):
        if token_id == "222":
            raise RuntimeError("simulated rejection")

    clob = FakeClob([_sampling_entry("0xabc")], {"111": _book()})
    clob.set_post_side_effect(fail_no_leg)
    gamma = FakeGamma([market])

    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    from poly03.making.execution import refresh_universe

    universe = refresh_universe(gamma, clob, max_gamma_markets=10)
    report = run_live_tick(state, universe=universe, gamma=gamma, clob=clob, max_markets_quoted=10, dry_run=False)

    # The bid leg placed successfully (report.placed records the attempt),
    # but since the ask leg failed it must not be left resting -- rolled
    # back rather than kept as a naked one-sided order.
    assert len(report.placed) == 1
    assert state.open_orders == []
    assert clob.cancelled  # the bid leg that succeeded got cancelled
    assert any("rolled back" in e for e in report.errors)


# --- affordability ranking (task item 1c) -----------------------------------


def test_rank_affordable_excludes_markets_over_budget(market_factory):
    m1 = market_factory("Cheap?", best_bid=0.48, best_ask=0.52)
    m1.id = "cheap"
    m2 = market_factory("Expensive?", best_bid=0.48, best_ask=0.52)
    m2.id = "expensive"

    cheap = _qm(m1, min_size=20, daily=1.0)  # low reward rate but affordable
    expensive = _qm(m2, min_size=1000, daily=1000.0)  # huge rate, unaffordable

    ranked = _rank_affordable([expensive, cheap], per_market_budget_usd=25.0)

    assert [qm.market.id for qm in ranked] == ["cheap"]


def test_rank_affordable_orders_by_reward_density_per_dollar(market_factory):
    m1 = market_factory("A?", best_bid=0.48, best_ask=0.52)
    m1.id = "a"
    m2 = market_factory("B?", best_bid=0.48, best_ask=0.52)
    m2.id = "b"

    low_density = _qm(m1, min_size=20, daily=5.0)  # $0.25/share
    high_density = _qm(m2, min_size=20, daily=20.0)  # $1.00/share

    ranked = _rank_affordable([low_density, high_density], per_market_budget_usd=25.0)

    assert [qm.market.id for qm in ranked] == ["b", "a"]


# --- markout windowing (task item 4) ----------------------------------------


class _FixedBookClob:
    def __init__(self, book):
        self.book = book

    def get_order_book(self, token_id):
        return self.book


def test_markout_scored_inside_window_missed_outside_it():
    book = OrderBook(asset_id="111", bids=[{"price": 0.50, "size": 100}], asks=[{"price": 0.52, "size": 100}])
    clob = _FixedBookClob(book)
    state = LiveMakingState()
    now = datetime.now(timezone.utc)

    on_time = state.record_fill(
        market_id="m", condition_id="c", token_id="111", question="q", side="buy", price=0.49, size_shares=20, order_id="o1"
    )
    on_time.filled_at = (now - timedelta(minutes=6)).isoformat()

    late = state.record_fill(
        market_id="m", condition_id="c", token_id="111", question="q", side="buy", price=0.49, size_shares=20, order_id="o2"
    )
    late.filled_at = (now - timedelta(minutes=20)).isoformat()

    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())
    compute_markouts(state, clob, report)

    assert on_time.markout_5m_usd is not None
    assert late.markout_5m_usd is None  # window (5-7min) was missed


# --- kill switches (task item 5) --------------------------------------------


def test_drawdown_kill_switch_trips_at_configured_fraction():
    from poly03.config import MAKING_LIVE_KILL_DRAWDOWN_FRACTION

    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    check_drawdown_kill_switch(state, report)
    assert not state.halted

    state.cash_usd = 100.0 * (1 - MAKING_LIVE_KILL_DRAWDOWN_FRACTION) - 0.01
    check_drawdown_kill_switch(state, report)
    assert state.halted
    assert any("drawdown kill switch" in r for r in state.halt_reasons)


def test_adverse_selection_kill_switch_still_requires_consecutive_bad_fills():
    from poly03.config import MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE

    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    for _ in range(MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE - 1):
        f = state.record_fill(
            market_id="m", condition_id="c", token_id="111", question="q", side="buy", price=0.50, size_shares=20, order_id="o"
        )
        f.markout_5m_usd = -10.0  # badly adverse
    check_adverse_selection_kill_switch(state, report)
    assert not state.halted  # not enough scored fills yet


# --- cancel-on-halt (task item 5) -------------------------------------------


def test_halted_tick_cancels_tracked_resting_orders(market_factory):
    from poly03.making.live_state import LiveOrder

    market = _quotable_market(market_factory)
    market.id = "test-1"
    gamma = FakeGamma([market])
    clob = FakeClob([_sampling_entry("0xabc")], {"111": _book()})

    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    state.halted = True
    state.halt_reasons.append("test halt")
    state.add_order(
        LiveOrder(
            order_id="order-1",
            market_id="test-1",
            condition_id="0xabc",
            token_id="111",
            question="Test?",
            side="buy",
            price=0.49,
            size_shares=20,
            quoted_midpoint=0.50,
        )
    )

    from poly03.making.execution import refresh_universe

    universe = refresh_universe(gamma, clob, max_gamma_markets=10)
    run_live_tick(state, universe=universe, gamma=gamma, clob=clob, max_markets_quoted=10, dry_run=False)

    assert clob.cancelled == [["order-1"]]
    assert state.open_orders == []
