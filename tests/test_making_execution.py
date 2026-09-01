from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from poly03.classifier.rules import Classification
from poly03.classifier.taxonomy import Tier
from poly03.config import MAKING_QUOTE_SIZE_MULTIPLE
from poly03.data.models import OrderBook
from poly03.making.execution import (
    _PHASE0_SIZING,
    _SHAKEDOWN_SIZING,
    LiveTickReport,
    _place_side,
    _rank_affordable,
    _sizing_fractions,
    check_adverse_selection_kill_switch,
    check_drawdown_kill_switch,
    check_market_pauses,
    cancel_stale_quotes,
    compute_markouts,
    reconcile_fills,
    reconcile_rewards,
    reconcile_trades,
    run_live_tick,
)
from poly03.making.live_state import LiveMakingState, LiveOrder
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
    def __init__(self, sampling=(), books=None, conditional_balances=None, trades=(), earnings=None):
        self._sampling = list(sampling)
        # {"YYYY-MM-DD": [{"condition_id": ..., "earnings": "1.5"}, ...]}
        self.earnings = dict(earnings or {})
        self.earnings_calls: list[str] = []
        self._books = books or {}
        self.posted: list[dict] = []
        self.cancelled: list[list[str]] = []
        self._post_side_effect = None
        # Defaults to "unlimited" (float("inf")) so existing tests that don't
        # care about on-chain balance capping are unaffected; pass a dict
        # {token_id: shares} to simulate a specific real balance.
        self._conditional_balances = conditional_balances or {}
        self.trades: list[dict] = list(trades or ())

    def get_conditional_balance(self, token_id):
        return self._conditional_balances.get(token_id, float("inf"))

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

    account_address = "0xfunder"

    def get_trades(self, *, after=None):
        return list(self.trades)

    def get_earnings_for_day(self, date):
        self.earnings_calls.append(date)
        return list(self.earnings.get(date, []))

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

    # Big enough that a `min_size * MAKING_QUOTE_SIZE_MULTIPLE` quote clears
    # the per-market inventory cap (0.25 of the cap at this bankroll) -- at
    # $100 nothing is affordable and no leg is ever attempted.
    state = LiveMakingState(bankroll_cap_usd=200.0, cash_usd=200.0)
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


# --- stale-order cancellation isn't gated on ranking or placement room -----
# (incident 2026-08-31: an order sat at a fixed price for ~19 minutes and 4
# fills while its market's mid ran 10.5c away from it, because the market had
# dropped out of `selected` and nothing else ever re-checked it for drift.)


def test_stale_order_cancelled_even_when_market_falls_out_of_selected(market_factory):
    market_a = _quotable_market(market_factory)
    market_a.id = "market-a"
    market_a.condition_id = "0xaaa"
    market_a.clob_token_ids = ["a-yes", "a-no"]

    market_b = _quotable_market(market_factory)
    market_b.id = "market-b"
    market_b.condition_id = "0xbbb"
    market_b.clob_token_ids = ["b-yes", "b-no"]

    clob = FakeClob(
        [
            _sampling_entry("0xaaa", daily=1000),  # ranks first
            _sampling_entry("0xbbb", daily=1),  # ranks last -- falls out at max_markets_quoted=1
        ],
        {"a-yes": _book(), "b-yes": _book(best_bid=0.48, best_ask=0.52)},
    )
    gamma = FakeGamma([market_a, market_b])

    state = LiveMakingState(bankroll_cap_usd=100_000.0, cash_usd=100_000.0)
    state.open_orders.append(
        LiveOrder(
            order_id="stale-order-b",
            market_id="market-b",
            condition_id="0xbbb",
            token_id="b-yes",
            question=market_b.question,
            side="buy",
            price=0.30,
            size_shares=20,
            quoted_midpoint=0.30,  # current book mid is 0.50 -- 20c of drift
        )
    )

    from poly03.making.execution import refresh_universe

    universe = refresh_universe(gamma, clob, max_gamma_markets=10)
    assert {qm.market.id for qm in universe.quotable} == {"market-a", "market-b"}

    run_live_tick(state, universe=universe, gamma=gamma, clob=clob, max_markets_quoted=1, dry_run=False)

    assert "stale-order-b" in [oid for batch in clob.cancelled for oid in batch]
    assert "stale-order-b" not in [o.order_id for o in state.open_orders]


def test_stale_order_cancelled_even_when_budget_exhausted_for_replacement(market_factory):
    market = _quotable_market(market_factory)
    market.id = "market-a"

    # A second quotable market holding a large *fresh* (non-stale) order --
    # kept quotable rather than dropped from the universe so it isn't
    # unwound before `deployed` is computed, the way an unaffordable/expired
    # market would be. Its collateral alone eats almost the whole bankroll.
    market_other = _quotable_market(market_factory, best_bid=0.88, best_ask=0.92)
    market_other.id = "market-other"
    market_other.condition_id = "0xother"
    market_other.clob_token_ids = ["other-yes", "other-no"]

    clob = FakeClob(
        [_sampling_entry("0xabc"), _sampling_entry("0xother")],
        {"111": _book(), "other-yes": _book(best_bid=0.88, best_ask=0.92)},
    )
    gamma = FakeGamma([market, market_other])

    # Bankroll is mostly committed to the other market, so there's no room
    # left to place a replacement quote here -- the stale order must still
    # get cancelled even though it can't be replaced this tick.
    state = LiveMakingState(bankroll_cap_usd=200.0, cash_usd=200.0)
    state.open_orders.append(
        LiveOrder(
            order_id="fresh-order-other",
            market_id="market-other",
            condition_id="0xother",
            token_id="other-yes",
            question=market_other.question,
            side="buy",
            price=0.90,
            size_shares=200,  # $180 of the $200 cap -- no room for a replacement
            quoted_midpoint=0.90,  # matches current book mid -- not stale
        )
    )
    state.open_orders.append(
        LiveOrder(
            order_id="stale-order-a",
            market_id="market-a",
            condition_id="0xabc",
            token_id="111",
            question=market.question,
            side="buy",
            price=0.30,
            size_shares=20,
            quoted_midpoint=0.30,  # current book mid is 0.50 -- 20c of drift
        )
    )

    from poly03.making.execution import refresh_universe

    universe = refresh_universe(gamma, clob, max_gamma_markets=10)
    report = run_live_tick(state, universe=universe, gamma=gamma, clob=clob, max_markets_quoted=10, dry_run=False)

    assert "stale-order-a" in [oid for batch in clob.cancelled for oid in batch]
    assert "stale-order-a" not in [o.order_id for o in state.open_orders]
    assert "fresh-order-other" in [o.order_id for o in state.open_orders]  # untouched, wasn't stale
    assert "live_budget_exhausted" in report.skipped


# --- sizing fractions (Phase 0 vs. shakedown) --------------------------
#
# Threshold is $10k, not $500: Phase 0's 0.02 fraction only clears real
# min_size (observed up to 200 shares == $200 collateral, see execution.py's
# _PHASE0_SIZING_THRESHOLD_USD comment) once the bankroll is in the multiple
# thousands. At $500 it produced a $10/market budget that cleared nothing --
# confirmed live by 3,275 consecutive would_place=0 ticks.


def test_sizing_fractions_below_10k_uses_shakedown_knobs():
    assert _sizing_fractions(100.0) == _SHAKEDOWN_SIZING
    assert _sizing_fractions(500.0) == _SHAKEDOWN_SIZING
    assert _sizing_fractions(1000.0) == _SHAKEDOWN_SIZING
    assert _sizing_fractions(9_999.99) == _SHAKEDOWN_SIZING


def test_sizing_fractions_at_or_above_10k_uses_phase0_defaults():
    assert _sizing_fractions(10_000.0) == _PHASE0_SIZING
    assert _sizing_fractions(50_000.0) == _PHASE0_SIZING


# --- the suite must never write to production run state ---------------------


def test_decision_logging_is_sandboxed_away_from_the_live_log():
    """A test run once appended ~80 fabricated placements and cancels to the
    real `making_live_decisions.jsonl`, because several tests call
    `run_live_tick`/`_place_side` without a `decision_log_path` and the
    module default is the production path. Those entries then read as real
    activity minutes after the live loop had actually stopped. conftest's
    autouse fixture intercepts the write; this asserts it is in force."""
    from pathlib import Path

    from poly03.config import MAKING_LIVE_DECISION_LOG_FILE
    from poly03.making.live_state import log_event

    live = Path(MAKING_LIVE_DECISION_LOG_FILE)
    before = live.stat().st_mtime_ns if live.exists() else None

    # Deliberately taking the default path -- the exact call shape that leaked.
    log_event({"kind": "place", "market_id": "should-never-reach-production"})

    after = live.stat().st_mtime_ns if live.exists() else None
    assert after == before, f"{MAKING_LIVE_DECISION_LOG_FILE} was written to by the test suite"


# --- affordability ranking (task item 1c) -----------------------------------


def test_rank_affordable_excludes_markets_over_budget(market_factory):
    m1 = market_factory("Cheap?", best_bid=0.48, best_ask=0.52)
    m1.id = "cheap"
    m2 = market_factory("Expensive?", best_bid=0.48, best_ask=0.52)
    m2.id = "expensive"

    cheap = _qm(m1, min_size=20, daily=1.0)  # low reward rate but affordable
    expensive = _qm(m2, min_size=1000, daily=1000.0)  # huge rate, unaffordable

    # A two-sided quote costs `min_size * MAKING_QUOTE_SIZE_MULTIPLE` dollars,
    # not `min_size` -- budget the cheap market in and the expensive one out.
    budget = 20 * MAKING_QUOTE_SIZE_MULTIPLE + 5.0
    ranked = _rank_affordable([expensive, cheap], per_market_budget_usd=budget)

    assert [qm.market.id for qm in ranked] == ["cheap"]


def test_rank_affordable_budgets_the_quoted_size_not_the_bare_minimum(market_factory):
    """A market affordable at bare min_size but not at the size we actually
    quote must not be ranked in -- it would only be skipped later by the
    inventory cap, displacing a market we could have quoted."""
    m = market_factory("Marginal?", best_bid=0.48, best_ask=0.52)
    m.id = "marginal"
    marginal = _qm(m, min_size=20, daily=10.0)

    just_under = 20 * MAKING_QUOTE_SIZE_MULTIPLE - 1.0
    assert _rank_affordable([marginal], per_market_budget_usd=just_under) == []
    assert _rank_affordable([marginal], per_market_budget_usd=just_under + 2.0) == [marginal]


def test_rank_affordable_orders_by_reward_density_per_dollar(market_factory):
    m1 = market_factory("A?", best_bid=0.48, best_ask=0.52)
    m1.id = "a"
    m2 = market_factory("B?", best_bid=0.48, best_ask=0.52)
    m2.id = "b"

    low_density = _qm(m1, min_size=20, daily=5.0)  # $0.25/share
    high_density = _qm(m2, min_size=20, daily=20.0)  # $1.00/share

    ranked = _rank_affordable(
        [low_density, high_density], per_market_budget_usd=20 * MAKING_QUOTE_SIZE_MULTIPLE + 5.0
    )

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
    # Equity is only actionable once trade history has confirmed the book;
    # see test_drawdown_kill_switch_does_not_halt_on_an_unconfirmed_book.
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport(), trades_reconciled=True)

    check_drawdown_kill_switch(state, report)
    assert not state.halted

    state.cash_usd = 100.0 * (1 - MAKING_LIVE_KILL_DRAWDOWN_FRACTION) - 0.01
    check_drawdown_kill_switch(state, report)
    assert state.halted
    assert any("drawdown kill switch" in r for r in state.halt_reasons)


def test_drawdown_kill_switch_does_not_halt_on_an_unconfirmed_book():
    """2026-08-31: the book halted at "equity $421.15 <= floor $425.00" while
    trade history put real equity at $479.72 -- the floor was never breached,
    the gap was phantom fills and fabricated fees. Equity moves with every
    mis-booked fill, so it is only worth halting on once this tick has
    actually reconciled against trade history."""
    from poly03.config import MAKING_LIVE_KILL_DRAWDOWN_FRACTION

    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0 * (1 - MAKING_LIVE_KILL_DRAWDOWN_FRACTION) - 0.01)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())
    assert report.trades_reconciled is False

    check_drawdown_kill_switch(state, report)

    assert not state.halted
    assert any("unreconciled" in e for e in report.errors)

    # ...and the moment the same equity *is* confirmed, it halts.
    report.trades_reconciled = True
    check_drawdown_kill_switch(state, report)
    assert state.halted


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


def test_resume_from_halt_does_not_immediately_redeadlock_on_the_same_fills():
    """2026-08-31 incident: clearing `halted` alone used to deadlock -- the
    kill switch re-evaluates the same trailing window every tick, so if
    nothing new has filled yet (guaranteed right after a resume), it saw the
    exact same bad fills that caused the halt and re-tripped before the book
    could earn a single new fill."""
    from poly03.config import MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE

    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    for _ in range(MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE):
        f = state.record_fill(
            market_id="m", condition_id="c", token_id="111", question="q", side="buy", price=0.50, size_shares=20, order_id="o"
        )
        f.markout_5m_usd = -10.0
    check_adverse_selection_kill_switch(state, report)
    assert state.halted

    state.resume_from_halt()
    assert not state.halted

    # A tick runs right after resume, before any new fill lands -- the same
    # stale fills are still the "last N scored", but resume acked them.
    check_adverse_selection_kill_switch(state, report)
    assert not state.halted

    # A fresh run of N bad fills after the ack point still trips it.
    for _ in range(MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE):
        f = state.record_fill(
            market_id="m", condition_id="c", token_id="111", question="q", side="buy", price=0.50, size_shares=20, order_id="o2"
        )
        f.markout_5m_usd = -10.0
    check_adverse_selection_kill_switch(state, report)
    assert state.halted


def test_market_pause_trips_on_that_markets_own_bad_streak_without_halting_book():
    from poly03.config import MAKING_LIVE_MARKET_PAUSE_CONSECUTIVE

    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    for _ in range(MAKING_LIVE_MARKET_PAUSE_CONSECUTIVE):
        f = state.record_fill(
            market_id="bad-market", condition_id="c", token_id="111", question="q",
            side="buy", price=0.50, size_shares=20, order_id="o",
        )
        f.markout_5m_usd = -10.0  # badly adverse

    check_market_pauses(state, report)

    assert "bad-market" in state.paused_markets
    assert not state.halted  # book-wide halt is a separate, higher bar
    assert any("market pause" in e for e in report.errors)


def test_market_pause_does_not_trip_other_markets():
    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    from poly03.config import MAKING_LIVE_MARKET_PAUSE_CONSECUTIVE

    for _ in range(MAKING_LIVE_MARKET_PAUSE_CONSECUTIVE):
        f = state.record_fill(
            market_id="bad-market", condition_id="c", token_id="111", question="q",
            side="buy", price=0.50, size_shares=20, order_id="o",
        )
        f.markout_5m_usd = -10.0
    good = state.record_fill(
        market_id="good-market", condition_id="c2", token_id="222", question="q2",
        side="buy", price=0.50, size_shares=20, order_id="o2",
    )
    good.markout_5m_usd = 5.0

    check_market_pauses(state, report)

    assert "bad-market" in state.paused_markets
    assert "good-market" not in state.paused_markets


def test_market_pause_clears_once_a_later_fill_scores_within_threshold():
    from poly03.config import MAKING_LIVE_MARKET_PAUSE_CONSECUTIVE

    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    for _ in range(MAKING_LIVE_MARKET_PAUSE_CONSECUTIVE):
        f = state.record_fill(
            market_id="m", condition_id="c", token_id="111", question="q",
            side="buy", price=0.50, size_shares=20, order_id="o",
        )
        f.markout_5m_usd = -10.0
    check_market_pauses(state, report)
    assert "m" in state.paused_markets

    good = state.record_fill(
        market_id="m", condition_id="c", token_id="111", question="q",
        side="buy", price=0.50, size_shares=20, order_id="o2",
    )
    good.markout_5m_usd = 5.0
    check_market_pauses(state, report)
    assert "m" not in state.paused_markets


# --- leaving the universe is not a reason to pay the spread -----------------
#
# Unwinding on *any* universe drop turned a wobble across a threshold (24h
# volume near $1k, spread tightening to a tick, a reward-rate reshuffle) into
# a forced taker sale. It shows in the trade history as synchronised
# liquidation bursts on the refresh boundary (2026-09-01: 05:20, 08:23,
# 08:53), median hold 58 minutes. Only a deadline should unwind now.


def _state_holding(market_id="gone", token_id="111", **pos_kw):
    from poly03.making.live_state import LiveInventory, LiveOrder

    state = LiveMakingState(bankroll_cap_usd=200.0, cash_usd=200.0)
    state.positions.append(
        LiveInventory(
            market_id=market_id, condition_id="0xgone", token_id=token_id,
            question="q", net_shares=20.0, avg_price=0.49, **pos_kw
        )
    )
    state.add_order(
        LiveOrder(
            order_id="resting", market_id=market_id, condition_id="0xgone",
            token_id=token_id, question="q", side="buy", price=0.49,
            size_shares=20, quoted_midpoint=0.50,
        )
    )
    return state


def test_market_leaving_the_universe_cancels_quotes_but_holds_inventory():
    clob = FakeClob(books={"111": _book(best_bid=0.48, best_ask=0.52)})
    state = _state_holding()

    report = run_live_tick(
        state, universe=UniverseReport(), gamma=FakeGamma([]), clob=clob,
        max_markets_quoted=10, dry_run=False,
    )

    assert clob.posted == []  # nothing sold
    assert state.open_orders == []  # but the quote is pulled
    assert state.positions[0].net_shares == 20.0
    assert "left_universe_holding_inventory" in report.skipped


def test_market_inside_the_flatten_window_still_unwinds():
    clob = FakeClob(books={"111": _book(best_bid=0.48, best_ask=0.52)})
    state = _state_holding()

    universe = UniverseReport()
    universe.require_unwind("gone", "resolving_within_flatten_window")

    run_live_tick(
        state, universe=universe, gamma=FakeGamma([]), clob=clob,
        max_markets_quoted=10, dry_run=False,
    )

    assert len(clob.posted) == 1
    assert clob.posted[0]["side"] == "SELL"


def test_a_position_past_its_own_deadline_unwinds_even_if_never_rescanned():
    """select_universe breaks at the 24h-volume floor, so a market that goes
    quiet is never scanned again and can never be reported as resolving. The
    deadline stamped on the position is what stops it being held into
    resolution anyway."""
    clob = FakeClob(books={"111": _book(best_bid=0.48, best_ask=0.52)})
    soon = (datetime.now(timezone.utc) + timedelta(hours=6)).isoformat()
    state = _state_holding(end_date_iso=soon)

    run_live_tick(
        state, universe=UniverseReport(), gamma=FakeGamma([]), clob=clob,
        max_markets_quoted=10, dry_run=False,
    )

    assert len(clob.posted) == 1
    assert clob.posted[0]["side"] == "SELL"


def test_a_distant_deadline_does_not_unwind():
    clob = FakeClob(books={"111": _book(best_bid=0.48, best_ask=0.52)})
    far = (datetime.now(timezone.utc) + timedelta(days=90)).isoformat()
    state = _state_holding(end_date_iso=far)

    run_live_tick(
        state, universe=UniverseReport(), gamma=FakeGamma([]), clob=clob,
        max_markets_quoted=10, dry_run=False,
    )

    assert clob.posted == []
    assert state.positions[0].net_shares == 20.0


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


# --- flatten dedup (2026-08-31 incident: "not enough balance / allowance") -


def test_flatten_position_tracks_order_and_skips_while_one_still_resting():
    """A flatten order used to be fire-and-forget: nothing recorded it as
    resting, so a second call (e.g. the next tick, before it fills or gets
    cancelled) placed a *second* full-size sell on top of the first. The
    exchange reserves shares against the first resting order, so the second
    request exceeds free balance and the CLOB rejects it with 'not enough
    balance / allowance'. Fixed by tracking the flatten order like any other
    LiveOrder and skipping if one is already resting for this token/side."""
    from poly03.making.execution import _flatten_position
    from poly03.making.live_state import LiveInventory

    clob = FakeClob(books={"111": _book(best_bid=0.48, best_ask=0.52)})
    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    pos = LiveInventory(market_id="m1", condition_id="0xabc", token_id="111", question="q", net_shares=20.0, avg_price=0.49)
    state.positions.append(pos)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    _flatten_position(state, clob, pos, report, dry_run=False, decision_log_path="/dev/null")
    assert len(clob.posted) == 1
    assert clob.posted[0]["side"] == "SELL"
    assert len(state.open_orders) == 1
    assert state.open_orders[0].side == "sell"

    # Second call, nothing cancelled the resting flatten order in between --
    # must not place a duplicate.
    _flatten_position(state, clob, pos, report, dry_run=False, decision_log_path="/dev/null")
    assert len(clob.posted) == 1
    assert len(state.open_orders) == 1


def test_flatten_market_cancels_stale_flatten_order_before_reflattening():
    """_flatten_market already cancels every tracked order for the market
    before flattening positions -- once the flatten order is tracked (this
    fix), that existing cancel-then-place cycle naturally reissues it fresh
    each tick instead of stacking a duplicate."""
    from poly03.making.execution import _flatten_market
    from poly03.making.live_state import LiveInventory

    clob = FakeClob(books={"111": _book(best_bid=0.48, best_ask=0.52)})
    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    pos = LiveInventory(market_id="m1", condition_id="0xabc", token_id="111", question="q", net_shares=20.0, avg_price=0.49)
    state.positions.append(pos)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    _flatten_market(state, clob, "m1", report, dry_run=False, decision_log_path="/dev/null")
    assert len(clob.posted) == 1
    first_order_id = state.open_orders[0].order_id

    # Simulate the next tick, still unfilled: cancel + reflatten, not stack.
    _flatten_market(state, clob, "m1", report, dry_run=False, decision_log_path="/dev/null")
    assert clob.cancelled == [[first_order_id]]
    assert len(clob.posted) == 2
    assert len(state.open_orders) == 1


def test_flatten_position_caps_sell_size_to_real_on_chain_balance():
    """The dedup fix alone wasn't enough: state resumed from before the fix
    already had `net_shares` overstated vs. the real on-chain balance (a
    flatten fill from an untracked order was never reconciled -- a market
    that's left the quotable universe can't be adopted back, see
    reconcile_fills/_adopt_untracked_order), so the very first flatten
    attempt on a fresh process still asked to sell more than is actually
    held and got the same 'not enough balance / allowance' rejection. The
    real on-chain balance must cap what gets sold, regardless of what local
    bookkeeping claims."""
    from poly03.making.execution import _flatten_position
    from poly03.making.live_state import LiveInventory

    # Tracked position claims 90 shares; the chain only actually holds 40 --
    # exactly the drift shape from the incident.
    clob = FakeClob(books={"111": _book(best_bid=0.48, best_ask=0.52)}, conditional_balances={"111": 40.0})
    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    pos = LiveInventory(market_id="m1", condition_id="0xabc", token_id="111", question="q", net_shares=90.0, avg_price=0.49)
    state.positions.append(pos)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    _flatten_position(state, clob, pos, report, dry_run=False, decision_log_path="/dev/null")

    assert len(clob.posted) == 1
    assert clob.posted[0]["size"] == 40.0
    assert state.open_orders[0].size_shares == 40.0
    assert any("exceeds on-chain balance" in e for e in report.errors)
    # net_shares must be pulled down to on-chain truth, or every future tick
    # re-detects the same 90-vs-40 drift and logs the same error forever.
    assert pos.net_shares == 40.0


def test_flatten_position_skips_when_no_real_balance_remains():
    """If the chain reports ~0 (all of the tracked shares were already sold
    by an untracked order before this fix existed), there's nothing left to
    flatten -- must not attempt a zero/negative-size order."""
    from poly03.making.execution import _flatten_position
    from poly03.making.live_state import LiveInventory

    clob = FakeClob(books={"111": _book(best_bid=0.48, best_ask=0.52)}, conditional_balances={"111": 0.0})
    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    pos = LiveInventory(market_id="m1", condition_id="0xabc", token_id="111", question="q", net_shares=90.0, avg_price=0.49)
    state.positions.append(pos)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    _flatten_position(state, clob, pos, report, dry_run=False, decision_log_path="/dev/null")

    assert clob.posted == []
    assert state.open_orders == []
    # net_shares must collapse to 0 along with it, or this position keeps
    # re-triggering the same "nothing left to flatten" path -- and keeps
    # logging the same error -- on every tick forever, and keeps inflating
    # equity/deployed-collateral with shares that don't exist.
    assert pos.net_shares == 0.0
    assert state.open_positions == []


def test_flatten_position_writes_off_dust_below_minimum_tradable_size():
    """A leftover position smaller than the exchange's minimum expressible
    size (2 decimal places of a share -- see _MIN_TRADABLE_SHARES) rounds to
    a maker_amount of 0 in py_clob_client_v2's order builder, which the
    exchange rejects with 'invalid maker amount'. That's not transient, so
    retrying it every tick just repeats the same failure forever (2026-09-01
    incident: identical 'invalid maker amount' error, same token, every tick,
    indefinitely). Must write the dust off instead of attempting to sell it."""
    from poly03.making.execution import _flatten_position
    from poly03.making.live_state import LiveInventory

    clob = FakeClob(books={"111": _book(best_bid=0.48, best_ask=0.52)})
    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    pos = LiveInventory(market_id="m1", condition_id="0xabc", token_id="111", question="q", net_shares=0.004, avg_price=0.49)
    state.positions.append(pos)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    _flatten_position(state, clob, pos, report, dry_run=False, decision_log_path="/dev/null")

    assert clob.posted == []
    assert state.open_orders == []
    assert pos.net_shares == 0.0
    assert any("below the exchange's minimum tradable size" in e for e in report.errors)


# --- flatten prices passively before it crosses -----------------------------
#
# Crossing on the first tick is what made the exits cost -4.99c/share against
# a 2-4c gross spread (23 of 29 losing round trips, 2026-08-31/09-01). We are
# the maker on the way in; be the maker on the way out too, until the
# resolution deadline actually forces the issue.


def _long_position(net_shares=20.0, avg_price=0.49, **kw):
    from poly03.making.live_state import LiveInventory

    return LiveInventory(
        market_id="m1", condition_id="0xabc", token_id="111", question="q",
        net_shares=net_shares, avg_price=avg_price, **kw
    )


def test_flatten_rests_at_the_mid_instead_of_hitting_the_bid():
    from poly03.making.execution import _flatten_position

    clob = FakeClob(books={"111": _book(best_bid=0.48, best_ask=0.52)})
    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    pos = _long_position()
    state.positions.append(pos)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    _flatten_position(state, clob, pos, report, dry_run=False, decision_log_path="/dev/null")

    # Mid is 0.50; hitting the bid would have sold at 0.48.
    assert clob.posted[0]["price"] == pytest.approx(0.50)
    assert report.flattened[0]["reason"] == "flatten_passive_at_mid"
    assert pos.flatten_started_at is not None


def test_flatten_crosses_the_spread_once_the_passive_window_expires():
    from poly03.making.execution import _flatten_position

    clob = FakeClob(books={"111": _book(best_bid=0.48, best_ask=0.52)})
    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    stale = (datetime.now(timezone.utc) - timedelta(hours=4)).isoformat()
    pos = _long_position(flatten_started_at=stale)
    state.positions.append(pos)
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())

    _flatten_position(state, clob, pos, report, dry_run=False, decision_log_path="/dev/null")

    assert clob.posted[0]["price"] == pytest.approx(0.48)  # the touch
    assert report.flattened[0]["reason"] == "flatten_crossing_after_timeout"


def test_passive_flatten_never_rests_inside_or_across_the_touch():
    """A one-tick spread leaves no room between the bid and the mid, so the
    passive price must fall back to something that still rests as a maker
    rather than crossing."""
    from poly03.making.execution import _passive_flatten_price

    book = _book(best_bid=0.48, best_ask=0.49)
    price = _passive_flatten_price(book, "ask", 0.01, touch_price=0.48)
    assert price > 0.48  # not a taker sell into the bid
    assert price <= 0.49  # not resting past the offer


def test_flatten_timer_resets_once_the_position_goes_flat():
    state = LiveMakingState(bankroll_cap_usd=100.0, cash_usd=100.0)
    pos = _long_position(flatten_started_at="2020-01-01T00:00:00+00:00")
    state.positions.append(pos)

    state.record_fill(
        market_id="m1", condition_id="0xabc", token_id="111", question="q",
        side="sell", price=0.50, size_shares=20.0, order_id="o1",
    )

    assert pos.net_shares == 0.0
    assert pos.flatten_started_at is None


# --- fill reconciliation: matched size is booked exactly once (2026-08-31) ---
#
# The incident: an order's matched size was re-booked as new fills every time
# the order came back under management after being retired -- $81 of buys that
# never happened, a $56 gap between tracked and real equity, and a drawdown
# halt at an equity the account had never actually reached. See
# LiveMakingState.book_matched.


class _ReconcileClob:
    """Serves a fixed open-orders list, for reconcile_fills only."""

    def __init__(self, open_orders):
        self.open_orders = list(open_orders)

    def get_open_orders(self, **kw):
        return [dict(o) for o in self.open_orders]

    def get_order(self, order_id):
        for o in self.open_orders:
            if o["id"] == order_id:
                return dict(o)
        return None

    def get_order_book(self, token_id):
        return _book()

    def get_fee_rate_bps(self, token_id):
        return None

    def get_earnings_for_day(self, date):
        return []


def _remote_order(order_id="order-1", *, matched, size=20.0, price=0.45, status="LIVE"):
    return {
        "id": order_id,
        "asset_id": "111",
        "price": price,
        "original_size": size,
        "size_matched": matched,
        "status": status,
    }


def _resting(market_id, order_id="order-1", *, size=20.0, price=0.45):
    return LiveOrder(
        order_id=order_id,
        market_id=market_id,
        condition_id="0xabc",
        token_id="111",
        question="Test market?",
        side="buy",
        price=price,
        size_shares=size,
        quoted_midpoint=0.5,
    )


def _reconcile(state, clob, quotable, *, log_path):
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())
    reconcile_fills(state, clob, report, decision_log_path=log_path, token_to_market={"111": quotable})
    return report


def test_filled_order_still_listed_open_is_not_rebooked_every_tick(tmp_path, market_factory):
    """The exact incident shape: an order fills, gets retired locally, and the
    exchange keeps returning it from get_open_orders. Each later tick used to
    re-adopt it (size_matched back to what the payload said) and book its
    whole matched size again as a brand-new fill."""
    log_path = str(tmp_path / "decisions.jsonl")
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting(quotable.market.id))
    clob = _ReconcileClob([_remote_order(matched=20.0)])

    for _ in range(5):
        report = _reconcile(state, clob, quotable, log_path=log_path)

    assert len(state.fills) == 1
    assert state.fills[0].size_shares == 20.0
    assert state.positions[0].net_shares == 20.0
    assert state.cash_usd == 500.0 - 20.0 * 0.45
    # ...and it stops logging a drift error on every single tick, too.
    assert report.errors == []


def test_cancelled_order_that_the_exchange_still_lists_does_not_rebook(tmp_path, market_factory):
    """Order 0x2a3952b6's shape: cancelled locally after filling, then still
    present in later open-orders reads. Removal from state.open_orders must
    not amnesty the matched size it already booked."""
    log_path = str(tmp_path / "decisions.jsonl")
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting(quotable.market.id, size=40.0, price=0.72))
    clob = _ReconcileClob([_remote_order(matched=40.0, size=40.0, price=0.72)])

    _reconcile(state, clob, quotable, log_path=log_path)
    assert len(state.fills) == 1
    state.remove_order("order-1")  # e.g. a cancel the exchange never applied

    for _ in range(3):
        _reconcile(state, clob, quotable, log_path=log_path)

    assert len(state.fills) == 1
    assert state.positions[0].net_shares == 40.0


def test_readopted_order_books_only_the_matched_size_not_yet_booked(tmp_path, market_factory):
    """The ledger must not overshoot in the other direction: an order that
    matches *more* after we lose track of it still owes us that fill, and
    only that fill."""
    log_path = str(tmp_path / "decisions.jsonl")
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting(quotable.market.id, size=40.0, price=0.72))
    clob = _ReconcileClob([_remote_order(matched=15.0, size=40.0, price=0.72)])

    _reconcile(state, clob, quotable, log_path=log_path)
    assert [f.size_shares for f in state.fills] == [15.0]

    state.remove_order("order-1")
    clob.open_orders = [_remote_order(matched=40.0, size=40.0, price=0.72)]
    _reconcile(state, clob, quotable, log_path=log_path)

    assert [f.size_shares for f in state.fills] == [15.0, 25.0]
    assert state.positions[0].net_shares == 40.0


def test_untracked_order_with_matched_size_is_booked_once_when_adopted(tmp_path, market_factory):
    """An order we never tracked (lost post response) that already has a fill
    on it: adopting must book that fill once, not swallow it and not repeat
    it once the ledger knows about it."""
    log_path = str(tmp_path / "decisions.jsonl")
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = _ReconcileClob([_remote_order(matched=20.0, size=40.0)])

    for _ in range(3):
        _reconcile(state, clob, quotable, log_path=log_path)

    assert [f.size_shares for f in state.fills] == [20.0]
    assert state.positions[0].net_shares == 20.0


def test_booked_matched_survives_a_save_load_round_trip(tmp_path, market_factory):
    """The ledger is only useful if it outlives the process -- the run loop
    restarts constantly (Ctrl+C, halts, crashes) and a fresh state that has
    forgotten what it booked re-books everything still on the book."""
    from poly03.making.live_state import load_state, save_state

    log_path = str(tmp_path / "decisions.jsonl")
    state_path = tmp_path / "live_state.json"
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting(quotable.market.id))
    clob = _ReconcileClob([_remote_order(matched=20.0)])
    _reconcile(state, clob, quotable, log_path=log_path)
    save_state(state, state_path)

    resumed = load_state(state_path)
    assert resumed.booked_matched == {"order-1": 20.0}
    _reconcile(resumed, clob, quotable, log_path=log_path)

    assert len(resumed.fills) == 1
    assert resumed.cash_usd == 500.0 - 20.0 * 0.45


# --- fills booked from trade history (2026-08-31) ---------------------------
#
# reconcile_fills can only see a fill while the order is still tracked. 11
# real fills ($95.65) landed outside that window and were never booked --
# including all three buys of the market that lost the most money, which the
# inventory cap, the kill switch and the flatten path were all blind to
# because the position didn't exist as far as the state was concerned.

_FUNDER = "0xFuNdEr0000000000000000000000000000000001"


class _TradeClob(_ReconcileClob):
    account_address = _FUNDER

    def __init__(self, trades=()):
        super().__init__([])
        self.trades = list(trades)
        self.trade_calls: list[int | None] = []

    def get_trades(self, *, after=None):
        self.trade_calls.append(after)
        return [dict(t) for t in self.trades]


def _ago(minutes):
    return datetime.now(timezone.utc) - timedelta(minutes=minutes)


def _maker_trade(
    order_id="order-1",
    *,
    token_id="111",
    our_side="BUY",
    price=0.45,
    size=20.0,
    minutes_ago=10.0,
    condition_id="0xabc",
    fee_bps="",
    maker_address=_FUNDER,
    status="CONFIRMED",
):
    """A trade where we were the maker. Note the top-level side/price belong
    to the *taker* and are deliberately the opposite of ours -- reading them
    instead of our maker leg is the mistake this shape exists to catch."""
    taker_side = "SELL" if our_side == "BUY" else "BUY"
    return {
        "id": f"trade-{order_id}-{minutes_ago}",
        "market": condition_id,
        "asset_id": token_id,
        "side": taker_side,
        "size": str(size),
        "price": str(price),
        "status": status,
        "match_time": str(int(_ago(minutes_ago).timestamp())),
        "trader_side": "MAKER",
        "maker_orders": [
            {
                "order_id": order_id,
                "maker_address": maker_address,
                "matched_amount": str(size),
                "price": str(price),
                "asset_id": token_id,
                "side": our_side,
                "fee_rate_bps": fee_bps,
            }
        ],
    }


def _taker_trade(order_id="order-t", *, token_id="111", side="SELL", price=0.60, size=20.0, minutes_ago=10.0):
    return {
        "id": f"trade-{order_id}",
        "market": "0xabc",
        "asset_id": token_id,
        "side": side,
        "size": str(size),
        "price": str(price),
        "status": "CONFIRMED",
        "match_time": str(int(_ago(minutes_ago).timestamp())),
        "trader_side": "TAKER",
        "taker_order_id": order_id,
        "fee_rate_bps": "0",
        "maker_orders": [],
    }


def _reconcile_trades(state, clob, quotable, *, log_path):
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())
    token_to_market = {"111": quotable} if quotable is not None else {}
    reconcile_trades(state, clob, report, decision_log_path=log_path, token_to_market=token_to_market)
    return report


def test_maker_leg_is_booked_on_our_side_not_the_takers(tmp_path, market_factory):
    """The payload is taker-centric: our BUY is reported as a top-level SELL.
    Booking the top-level side would invert the entire book."""
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = _TradeClob([_maker_trade(our_side="BUY", price=0.45, size=20.0)])

    _reconcile_trades(state, clob, quotable, log_path=str(tmp_path / "d.jsonl"))

    assert [(f.side, f.size_shares, f.price) for f in state.fills] == [("buy", 20.0, 0.45)]
    assert state.positions[0].net_shares == 20.0
    assert state.cash_usd == 500.0 - 20.0 * 0.45


def test_fill_is_booked_even_though_no_order_was_ever_tracked(tmp_path, market_factory):
    """The incident's core failure: the fill landed after the order stopped
    being tracked (cancel-on-shutdown, a crash, a market leaving the
    universe), so order-diff reconciliation could never see it."""
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    assert state.open_orders == []
    clob = _TradeClob([_maker_trade(our_side="BUY", price=0.50, size=20.0)])

    _reconcile_trades(state, clob, None, log_path=str(tmp_path / "d.jsonl"))

    assert [f.size_shares for f in state.fills] == [20.0]
    assert state.open_positions[0].net_shares == 20.0


def test_trades_are_not_rebooked_on_every_tick(tmp_path, market_factory):
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = _TradeClob([_maker_trade(our_side="BUY", price=0.45, size=20.0)])

    for _ in range(5):
        _reconcile_trades(state, clob, quotable, log_path=str(tmp_path / "d.jsonl"))

    assert len(state.fills) == 1
    assert state.cash_usd == 500.0 - 20.0 * 0.45


def test_trade_reconciliation_does_not_rebook_what_order_diff_already_booked(tmp_path, market_factory):
    """Both sources share state.booked_matched, keyed by order id -- whichever
    sees a fill first books it and the other no-ops. Without that they'd each
    book the same fill independently."""
    log_path = str(tmp_path / "d.jsonl")
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting(quotable.market.id))
    order_clob = _ReconcileClob([_remote_order(matched=20.0)])
    _reconcile(state, order_clob, quotable, log_path=log_path)
    assert len(state.fills) == 1

    trade_clob = _TradeClob([_maker_trade("order-1", our_side="BUY", price=0.45, size=20.0)])
    _reconcile_trades(state, trade_clob, quotable, log_path=log_path)

    assert len(state.fills) == 1
    assert state.positions[0].net_shares == 20.0


def test_only_the_unbooked_slice_is_taken_and_at_its_own_price(tmp_path, market_factory):
    """An order that filled twice at different prices, with only the first
    already booked: the second must be booked at *its* price, not at a blend
    of the two."""
    log_path = str(tmp_path / "d.jsonl")
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.book_matched("order-1", 20.0)  # first leg already booked elsewhere
    clob = _TradeClob(
        [
            _maker_trade("order-1", our_side="BUY", price=0.50, size=20.0, minutes_ago=30),
            _maker_trade("order-1", our_side="BUY", price=0.36, size=15.0, minutes_ago=10),
        ]
    )

    _reconcile_trades(state, clob, quotable, log_path=log_path)

    assert [(f.size_shares, f.price) for f in state.fills] == [(15.0, 0.36)]


def test_counterparty_maker_legs_are_not_attributed_to_us(tmp_path, market_factory):
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = _TradeClob([_maker_trade(our_side="BUY", maker_address="0xSomeoneElse")])

    _reconcile_trades(state, clob, quotable, log_path=str(tmp_path / "d.jsonl"))

    assert state.fills == []
    assert state.cash_usd == 500.0


def test_failed_trades_are_ignored(tmp_path, market_factory):
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = _TradeClob([_maker_trade(our_side="BUY", status="FAILED")])

    _reconcile_trades(state, clob, quotable, log_path=str(tmp_path / "d.jsonl"))

    assert state.fills == []


def test_taker_leg_reduces_inventory(tmp_path, market_factory):
    """Flatten orders cross the spread, so they come back as TAKER trades --
    where the top-level fields *are* ours."""
    log_path = str(tmp_path / "d.jsonl")
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = _TradeClob(
        [
            _maker_trade("order-1", our_side="BUY", price=0.45, size=20.0, minutes_ago=30),
            _taker_trade("order-2", side="SELL", price=0.60, size=20.0, minutes_ago=10),
        ]
    )

    _reconcile_trades(state, clob, quotable, log_path=log_path)

    assert [(f.side, f.size_shares) for f in state.fills] == [("buy", 20.0), ("sell", 20.0)]
    assert state.open_positions == []
    assert state.cash_usd == 500.0 - 20.0 * 0.45 + 20.0 * 0.60


def test_fill_is_stamped_with_the_trades_own_match_time(tmp_path, market_factory):
    """A fill surfaced late must not claim to have happened now -- markout
    windows key off filled_at, and a wrong timestamp would score a 40-minute-
    old fill as if it were a fresh 5-minute markout."""
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = _TradeClob([_maker_trade(our_side="BUY", minutes_ago=40.0)])

    _reconcile_trades(state, clob, quotable, log_path=str(tmp_path / "d.jsonl"))

    age = (datetime.now(timezone.utc) - datetime.fromisoformat(state.fills[0].filled_at)).total_seconds()
    assert 39 * 60 < age < 41 * 60
    # ...and no mid is invented for it; "the mid now" is not the mid then.
    assert state.fills[0].mid_price_at_fill == 0.0


def test_existing_position_row_wins_so_inventory_never_splits(tmp_path, market_factory):
    """Fills key inventory on (market_id, token_id). If the same token
    resolved to a different market_id later in its life, one real position
    would be split across two rows that never net against each other."""
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.record_fill(
        market_id="legacy-market",
        condition_id="0xabc",
        token_id="111",
        question="Test market?",
        side="buy",
        price=0.40,
        size_shares=10.0,
        order_id="older-order",
    )
    clob = _TradeClob([_maker_trade("order-1", token_id="111", our_side="BUY", price=0.45, size=20.0)])

    _reconcile_trades(state, clob, quotable, log_path=str(tmp_path / "d.jsonl"))

    assert len(state.positions) == 1
    assert state.positions[0].market_id == "legacy-market"
    assert state.positions[0].net_shares == 30.0


def test_unresolvable_market_still_books_the_fill_against_its_condition_id(tmp_path, market_factory):
    """Nothing knows this token -- not the universe, not a position, not a
    tracked order. Dropping the fill was the old behaviour; the money moved
    either way, so book it against the one identifier the trade always
    carries."""
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = _TradeClob([_maker_trade(token_id="999", our_side="BUY", price=0.30, size=20.0, condition_id="0xdead")])

    _reconcile_trades(state, clob, None, log_path=str(tmp_path / "d.jsonl"))

    assert state.positions[0].market_id == "0xdead"
    assert state.positions[0].net_shares == 20.0


def test_trade_fees_come_from_the_trade_not_the_markets_posted_rate(tmp_path, market_factory):
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = _TradeClob([_maker_trade(our_side="BUY", price=0.50, size=20.0, fee_bps="100")])

    _reconcile_trades(state, clob, quotable, log_path=str(tmp_path / "d.jsonl"))

    assert state.realized_fee_usd_total == 0.01 * 0.50 * 20.0
    assert state.cash_usd == 500.0 - 20.0 * 0.50 - 0.10


def test_trade_history_is_requested_with_a_bounded_lookback(tmp_path, market_factory):
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = _TradeClob([])

    _reconcile_trades(state, clob, quotable, log_path=str(tmp_path / "d.jsonl"))

    now = datetime.now(timezone.utc).timestamp()
    assert clob.trade_calls and now - 4 * 86400 < clob.trade_calls[0] <= now - 2 * 86400


# --- trade history as the primary fill source (2026-08-31) -------------------
#
# The incident's P&L was wrong in three ways at once: fills booked twice, real
# fills never booked at all, and a fee line fabricated from the market's posted
# ceiling rather than the rate actually charged. Trade history fixes all three
# because it is the exchange's own record of what matched -- it does not depend
# on us still tracking the order, and it carries the real price and fee.


class _PostedRateClob(_ReconcileClob):
    """Order-tracking source whose market advertises a fat posted fee rate.
    1000bps is not hypothetical -- it is what `get_fee_rate_bps` returned for
    three of the markets quoted on 2026-08-31."""

    def __init__(self, open_orders):
        super().__init__(open_orders)
        self.fee_rate_calls: list[str] = []

    def get_fee_rate_bps(self, token_id):
        self.fee_rate_calls.append(token_id)
        return 1000


def test_order_path_books_no_fee_from_the_markets_posted_rate(tmp_path, market_factory):
    """The posted rate is a ceiling, not a bill. Trade history showed all 30
    legs of the incident charged zero while the book accrued $8.86 -- 44% of a
    $20.28 loss that never happened. The order path cannot know a fee, so it
    must not invent one."""
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting("m", size=20.0, price=0.45))
    clob = _PostedRateClob([_remote_order(matched=20.0, size=20.0, price=0.45, status="MATCHED")])

    _reconcile(state, clob, quotable, log_path=str(tmp_path / "d.jsonl"))

    assert len(state.fills) == 1
    assert state.fills[0].fee_usd == 0.0
    assert state.realized_fee_usd_total == 0.0
    # Cash moves by the notional and nothing else.
    assert state.cash_usd == 500.0 - 20.0 * 0.45
    # And we no longer even ask -- the endpoint cannot answer the question.
    assert clob.fee_rate_calls == []


def test_fills_record_which_reconciler_booked_them(tmp_path, market_factory):
    quotable = _qm(_quotable_market(market_factory))

    from_trades = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    _reconcile_trades(
        from_trades,
        _TradeClob([_maker_trade(our_side="BUY", price=0.45, size=20.0)]),
        quotable,
        log_path=str(tmp_path / "t.jsonl"),
    )
    assert from_trades.fills[0].source == "trades"

    from_orders = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    from_orders.add_order(_resting("m", size=20.0, price=0.45))
    _reconcile(
        from_orders,
        _ReconcileClob([_remote_order(matched=20.0, status="MATCHED")]),
        quotable,
        log_path=str(tmp_path / "o.jsonl"),
    )
    assert from_orders.fills[0].source == "orders"


def test_trade_history_wins_the_ledger_race_against_order_tracking(tmp_path, market_factory):
    """Both reconcilers can see the same fill. Whichever books it first owns
    the `booked_matched` ledger and the other no-ops -- so the ordering in
    run_live_tick is what decides whether the fill carries the real fee and
    price from trade history, or the order path's fee-free approximation.
    Trade history must go first."""
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting("m", size=20.0, price=0.45))

    # Same 20 shares, visible to both sources at once.
    _reconcile_trades(
        state,
        _TradeClob([_maker_trade(order_id="order-1", our_side="BUY", price=0.45, size=20.0, fee_bps="30")]),
        quotable,
        log_path=str(tmp_path / "d.jsonl"),
    )
    _reconcile(state, _PostedRateClob([_remote_order(matched=20.0, status="MATCHED")]), quotable, log_path=str(tmp_path / "d.jsonl"))

    assert len(state.fills) == 1, "the same 20 shares must not be booked twice"
    assert state.fills[0].source == "trades"
    assert state.fills[0].fee_usd == 0.003 * 0.45 * 20.0
    assert state.positions[0].net_shares == 20.0


def test_run_live_tick_reconciles_trades_before_orders(market_factory, monkeypatch):
    """Ordering is load-bearing (see the test above), so assert it directly
    rather than trusting the two calls stay in the right order."""
    import poly03.making.execution as ex

    calls: list[str] = []
    monkeypatch.setattr(ex, "reconcile_trades", lambda *a, **k: calls.append("trades"))
    monkeypatch.setattr(ex, "reconcile_fills", lambda *a, **k: calls.append("fills"))
    monkeypatch.setattr(ex, "compute_markouts", lambda *a, **k: None)
    monkeypatch.setattr(ex, "_mark_to_market", lambda *a, **k: None)

    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    ex.run_live_tick(
        state,
        universe=UniverseReport(),
        gamma=FakeGamma(),
        clob=FakeClob(),
        max_markets_quoted=1,
        dry_run=False,
    )

    assert calls == ["trades", "fills"]


class _NoTradeHistoryClob(FakeClob):
    def get_trades(self, *, after=None):
        raise RuntimeError("trade history unavailable")


def test_unreconciled_tick_places_no_new_quotes(market_factory):
    """A book we could not confirm against trade history is a book whose
    positions, cluster exposure and deployed collateral are all unverified.
    Quoting into that is how the incident kept laddering into a market whose
    real position it could not see. Unlike a halt this clears itself as soon
    as get_trades answers again."""
    market = _quotable_market(market_factory)
    universe = UniverseReport(quotable=[_qm(market)])
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = _NoTradeHistoryClob(books={"111": _book(), "222": _book()})

    report = run_live_tick(
        state, universe=universe, gamma=FakeGamma(), clob=clob, max_markets_quoted=10, dry_run=False
    )

    assert clob.posted == []
    assert report.skipped.get("trades_unreconciled_no_new_quotes") == 1
    assert not state.halted, "an unreadable tick is transient, not a halt"


def test_a_fresh_book_does_not_adopt_the_wallets_earlier_trades(tmp_path, market_factory):
    """A reset book has an empty `booked_matched`, so without a floor every
    trade inside the lookback window reads as brand new and lands on the
    clean state -- re-booking days of unrelated wallet activity as Book M's
    own fills, with its cash."""
    quotable = _qm(_quotable_market(market_factory))
    older = _maker_trade(order_id="before-reset", our_side="BUY", price=0.45, size=20.0, minutes_ago=600.0)
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=480.65)
    state.trades_booked_through = (datetime.now(timezone.utc) - timedelta(minutes=5)).timestamp()
    clob = _TradeClob([older])

    _reconcile_trades(state, clob, quotable, log_path=str(tmp_path / "d.jsonl"))

    assert state.fills == []
    assert state.cash_usd == 480.65
    # The floor is what's asked of the API, not a post-filter.
    assert clob.trade_calls[0] == int(state.trades_booked_through)


def test_the_floor_never_shortens_the_lookback_for_an_established_book(tmp_path, market_factory):
    """`trades_booked_through` is a floor for books that have no history, not
    a rolling watermark -- an established book must still see the full
    lookback so a fill that surfaces late is not skipped."""
    quotable = _qm(_quotable_market(market_factory))
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    assert state.trades_booked_through is None
    clob = _TradeClob([_maker_trade(our_side="BUY", price=0.45, size=20.0, minutes_ago=600.0)])

    _reconcile_trades(state, clob, quotable, log_path=str(tmp_path / "d.jsonl"))

    assert len(state.fills) == 1


# --- liquidity rewards: the one positive term, finally measured -------------


def _earn(cond, amount):
    return {"condition_id": cond, "earnings": str(amount), "maker_address": _FUNDER}


def _reconcile_rewards(state, clob):
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())
    reconcile_rewards(state, clob, report)
    return report


def test_rewards_are_summed_across_the_days_markets():
    today = datetime.now(timezone.utc).date().isoformat()
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = FakeClob(earnings={today: [_earn("0xa", 0.32), _earn("0xb", 0.25), _earn("0xc", 0.001)]})

    _reconcile_rewards(state, clob)

    assert state.earned_rewards_by_day[today] == pytest.approx(0.571)
    assert state.earned_reward_usd_total == pytest.approx(0.571)


def test_rereading_a_growing_epoch_converges_instead_of_stacking():
    """The current day's epoch is still accumulating, so it gets re-read every
    tick. Accumulating rather than assigning would inflate rewards without
    bound -- the same double-booking shape as the fill incident, on the one
    number the strategy is judged by."""
    today = datetime.now(timezone.utc).date().isoformat()
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = FakeClob(earnings={today: [_earn("0xa", 1.0)]})

    _reconcile_rewards(state, clob)
    _reconcile_rewards(state, clob)
    clob.earnings[today] = [_earn("0xa", 2.5)]  # epoch grew
    _reconcile_rewards(state, clob)

    assert state.earned_reward_usd_total == pytest.approx(2.5)


def test_earned_rewards_never_move_cash():
    """An epoch is reported as earned before it settles. Crediting it to cash
    would push the book above the real wallet balance -- the drift class this
    whole incident was about."""
    today = datetime.now(timezone.utc).date().isoformat()
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=480.65)
    clob = FakeClob(earnings={today: [_earn("0xa", 2.88)]})

    _reconcile_rewards(state, clob)

    assert state.cash_usd == 480.65
    assert state.realized_reward_usd_total == 0.0
    assert state.earned_reward_usd_total == pytest.approx(2.88)


def test_a_day_with_no_rewards_is_recorded_as_zero_not_skipped():
    """Skipping empty days would let a stale figure survive after the
    exchange revises a day down."""
    today = datetime.now(timezone.utc).date().isoformat()
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.record_earned_rewards(today, 5.0)

    _reconcile_rewards(state, FakeClob(earnings={}))

    assert state.earned_rewards_by_day[today] == 0.0


def test_reward_fetch_failure_is_reported_not_fatal():
    class Broken(FakeClob):
        def get_earnings_for_day(self, date):
            raise RuntimeError("rewards endpoint down")

    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    report = _reconcile_rewards(state, Broken())

    assert state.earned_rewards_by_day == {}
    assert len(report.errors) == 2  # today + yesterday
    assert all("reconcile_rewards" in e for e in report.errors)


def test_run_live_tick_reconciles_rewards():
    today = datetime.now(timezone.utc).date().isoformat()
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    clob = FakeClob(earnings={today: [_earn("0xa", 1.25)]})

    report = run_live_tick(
        state, universe=UniverseReport(), gamma=FakeGamma(), clob=clob, max_markets_quoted=1, dry_run=False
    )

    assert state.earned_reward_usd_total == pytest.approx(1.25)
    assert not [e for e in report.errors if "reconcile_rewards" in e]


# --- stale-quote repricing (2026-08-31) -------------------------------------
#
# The staleness check itself predates the incident. The bug was where it
# lived: inside the per-market quoting loop, behind four `continue`s. So
# cancelling a stale quote -- which only sheds risk -- was gated on the engine
# being able to price a replacement, a far stronger condition.


def _one_sided_book(best_bid=0.60, size=500.0):
    """Bids only -- the ask side has been pulled."""
    return OrderBook(asset_id="111", bids=[{"price": best_bid, "size": size}], asks=[])


def _resting_quote(order_id="order-1", *, side="buy", token_id="111", price=0.72, quoted_mid=0.745):
    return LiveOrder(
        order_id=order_id,
        market_id="m",
        condition_id="0xabc",
        token_id=token_id,
        question="q",
        side=side,
        price=price,
        size_shares=40.0,
        quoted_midpoint=quoted_mid,
    )


def _sweep(state, clob):
    report = LiveTickReport(timestamp="t", dry_run=False, universe=UniverseReport())
    cancel_stale_quotes(state, clob, report, dry_run=False, decision_log_path="/dev/null")
    return report


def test_sweep_cancels_a_quote_the_mid_has_run_away_from():
    """The incident shape: a bid quoted against a 0.745 mid, still resting
    once the mid reached 0.640."""
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting_quote(quoted_mid=0.745))
    clob = FakeClob(books={"111": _book(best_bid=0.635, best_ask=0.645)})

    report = _sweep(state, clob)

    assert clob.cancelled == [["order-1"]]
    assert report.skipped.get("stale_quote_mid_moved") == 1
    assert state.open_orders == []


def test_sweep_cancels_when_the_book_cannot_be_fetched():
    """One `get_order_books` batch covers 100 tokens. Under the old code a
    single failed batch stranded every stale quote in that chunk for as long
    as it kept failing. A quote we cannot price is one we cannot manage."""

    class NoBooks(FakeClob):
        def get_order_books(self, token_ids):
            raise RuntimeError("book endpoint down")

    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting_quote())
    clob = NoBooks()

    report = _sweep(state, clob)

    assert clob.cancelled == [["order-1"]]
    assert report.skipped.get("stale_quote_no_reference_price") == 1
    assert any("stale-quote sweep" in e for e in report.errors)


def test_sweep_prices_against_the_surviving_side_of_a_one_sided_book():
    """The old loop skipped these markets outright (`book_not_two_sided`),
    which is precisely when a resting quote is most exposed. Best bid 0.60
    against a 0.745 quote is stale by 14.5c."""
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting_quote(quoted_mid=0.745))
    clob = FakeClob(books={"111": _one_sided_book(best_bid=0.60)})

    report = _sweep(state, clob)

    assert clob.cancelled == [["order-1"]]
    assert report.skipped.get("stale_quote_mid_moved") == 1


def test_sweep_leaves_a_fresh_quote_resting():
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting_quote(quoted_mid=0.745))
    clob = FakeClob(books={"111": _book(best_bid=0.742, best_ask=0.748)})

    report = _sweep(state, clob)

    assert clob.cancelled == []
    assert [o.order_id for o in state.open_orders] == ["order-1"]
    assert report.skipped == {}


def test_sweep_ignores_flatten_sells():
    """`_flatten_market` already cancels and re-places them every tick."""
    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting_quote(order_id="flat-1", side="sell"))
    clob = FakeClob(books={"111": _book(best_bid=0.635, best_ask=0.645)})

    report = _sweep(state, clob)

    assert clob.cancelled == []
    assert report.skipped == {}


def test_sweep_runs_before_the_unreconciled_tick_bails_out():
    """The unreconciled guard stops new quoting. It must not also strand the
    quotes already resting -- that would leave the book at its most exposed
    exactly when it knows least. Same argument for the halt path."""

    class NoTrades(FakeClob):
        def get_trades(self, *, after=None):
            raise RuntimeError("trade history unavailable")

    state = LiveMakingState(bankroll_cap_usd=500.0, cash_usd=500.0)
    state.add_order(_resting_quote(quoted_mid=0.745))
    clob = NoTrades(books={"111": _book(best_bid=0.635, best_ask=0.645)})

    report = run_live_tick(
        state, universe=UniverseReport(), gamma=FakeGamma(), clob=clob, max_markets_quoted=10, dry_run=False
    )

    assert report.skipped.get("stale_quote_mid_moved") == 1
    assert report.skipped.get("trades_unreconciled_no_new_quotes") == 1
    assert state.open_orders == []
