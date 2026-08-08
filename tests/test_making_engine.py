from __future__ import annotations

import pytest

from poly03.data.models import OrderBook
from poly03.making.engine import run_tick
from poly03.making.measurement import phase0_gate, reward_estimate, universe_funnel
from poly03.making.rewards import RewardConfig
from poly03.making.state import MakingState, MakingTickSummary
from poly03.making.universe import select_universe

QUOTABLE_DESC = (
    "The winner will be determined by the official election commission's "
    "certified results. This market will resolve based on a consensus of "
    "credible reporting."
)


class FakeGamma:
    def __init__(self, markets):
        self._markets = markets

    def iter_markets(self, **kwargs):
        yield from self._markets

    def get_event(self, event_id):
        raise AssertionError("get_event should not be called when market.event_id is unset in tests")


class FakeClob:
    def __init__(self, sampling, books):
        self._sampling = sampling
        self._books = books

    def iter_sampling_markets(self, *, max_markets=None):
        yield from self._sampling

    def get_order_books(self, token_ids):
        return {t: self._books[t] for t in token_ids if t in self._books}


def _sampling_entry(condition_id="0xabc", min_size=20, max_spread=4.5, daily=35, tick=0.01):
    return {
        "condition_id": condition_id,
        "minimum_tick_size": tick,
        "rewards": {"rates": [{"rewards_daily_rate": daily}], "min_size": min_size, "max_spread": max_spread},
    }


def _book(best_bid=0.48, best_ask=0.52, size=500.0):
    return OrderBook(
        asset_id="111",
        bids=[{"price": best_bid, "size": size}],
        asks=[{"price": best_ask, "size": size}],
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
    return market_factory("Which party will win the TX-34 House seat?", **params)


# --- §3.1 universe ----------------------------------------------------------


def test_select_universe_accepts_a_funded_liquid_market(market_factory):
    market = _quotable_market(market_factory)
    report = select_universe(FakeGamma([market]), FakeClob([_sampling_entry()], {}))
    assert len(report.quotable) == 1
    assert report.quotable[0].reward.daily_rate_usd == 35.0
    assert report.quotable[0].tick_size == 0.01


def test_market_without_funded_rewards_is_dropped(market_factory):
    report = select_universe(FakeGamma([_quotable_market(market_factory)]), FakeClob([], {}))
    assert report.quotable == []


@pytest.mark.parametrize(
    "overrides,reason",
    [
        ({"volume_24hr": 10.0}, "below_min_24h_volume"),
        ({"best_bid": 0.005, "best_ask": 0.015}, "price_outside_quotable_band"),
        ({"best_bid": 0.499, "best_ask": 0.501}, "spread_too_tight_to_improve"),
        ({"days_to_resolution": 0.5}, "resolving_within_flatten_window"),
        ({"accepting_orders": False}, "not_accepting_orders"),
    ],
)
def test_universe_gates(market_factory, overrides, reason):
    market = _quotable_market(market_factory, **overrides)
    report = select_universe(FakeGamma([market]), FakeClob([_sampling_entry()], {}))
    assert report.quotable == []
    assert reason in report.rejections


def test_wide_spread_is_kept_not_rejected(market_factory):
    """Book M must not inherit Book A's wide_spread filter -- a wide spread is
    the opportunity, not a defect (strategy_v2.md §3.1)."""
    market = _quotable_market(market_factory, best_bid=0.30, best_ask=0.70)
    report = select_universe(FakeGamma([market]), FakeClob([_sampling_entry()], {}))
    assert len(report.quotable) == 1


def test_standard_uma_boilerplate_does_not_reject(market_factory):
    """strategy_v2.md §1.2: 'consensus of credible reporting' is on ~70% of
    the venue and must not read as ambiguity."""
    market = _quotable_market(market_factory)
    assert "consensus of credible reporting" in market.description
    report = select_universe(FakeGamma([market]), FakeClob([_sampling_entry()], {}))
    assert "ambiguous_resolution_criteria" not in report.rejections


def test_tier_4_markets_are_quotable(market_factory):
    """Tier 4 means "requires a forecast", which is a claim about
    predictability, not resolution integrity. A coin-flip market is the *best*
    thing to quote, so Book M records the tier without gating on it. This
    corrects strategy_v2.md §3.5's assumption that the classifier carries over
    as an inventory-risk filter."""
    from poly03.classifier.taxonomy import Tier

    market = _quotable_market(market_factory, best_bid=0.48, best_ask=0.52)
    report = select_universe(FakeGamma([market]), FakeClob([_sampling_entry()], {}))

    assert len(report.quotable) == 1
    assert report.quotable[0].classification.tier == Tier.TIER_4
    assert "tier_4_excluded" not in report.rejections


def test_genuinely_ambiguous_wording_still_rejects(market_factory):
    market = _quotable_market(
        market_factory,
        description="Resolves to whatever outcome is widely reported, at the discretion of the market admin.",
    )
    report = select_universe(FakeGamma([market]), FakeClob([_sampling_entry()], {}))
    assert report.quotable == []
    assert "ambiguous_resolution_criteria" in report.rejections


# --- tick -------------------------------------------------------------------


def test_run_tick_produces_a_scored_observation(market_factory):
    market = _quotable_market(market_factory)
    clob = FakeClob([_sampling_entry()], {"111": _book()})
    state = MakingState(bankroll=100_000.0)

    report = run_tick(state, gamma=FakeGamma([market]), clob=clob, decision_log_path="/dev/null")

    assert len(report.observations) == 1
    obs = report.observations[0]
    assert obs.our_qscore > 0
    assert obs.competing_qscore > 0
    assert 0 < obs.share_fraction < 1
    assert obs.est_reward_usd_per_day == pytest.approx(obs.share_fraction * 35.0)
    assert obs.collateral_usd > 0
    assert state.n_ticks == 1


def test_run_tick_skips_when_min_size_exceeds_the_inventory_cap(market_factory):
    """§3.4: quoting under the venue's minimum scoring size earns nothing;
    quoting at it would breach the per-market cap. Neither is acceptable."""
    market = _quotable_market(market_factory)
    clob = FakeClob([_sampling_entry(min_size=10_000)], {"111": _book()})
    state = MakingState(bankroll=1_000.0)  # 2% cap = $20/market

    report = run_tick(state, gamma=FakeGamma([market]), clob=clob, decision_log_path="/dev/null")

    assert report.observations == []
    assert report.skipped.get("reward_min_size_exceeds_inventory_cap") == 1


def test_run_tick_respects_the_deployment_budget(market_factory):
    market = _quotable_market(market_factory)
    clob = FakeClob([_sampling_entry()], {"111": _book()})
    state = MakingState(bankroll=1.0)  # 60% of $1 buys nothing

    report = run_tick(state, gamma=FakeGamma([market]), clob=clob, decision_log_path="/dev/null")
    assert report.observations == []


# --- §4 measurement ---------------------------------------------------------


def _summary(est, collateral=1_000.0, pool=100.0):
    return MakingTickSummary(
        timestamp="2026-08-07T00:00:00+00:00",
        gamma_scanned=100,
        reward_eligible=50,
        quotable=10,
        quoted=5,
        total_collateral_usd=collateral,
        total_est_reward_usd_per_day=est,
        pool_usd_per_day_in_quoted_markets=pool,
    )


def test_reward_estimate_summarises_the_series():
    state = MakingState(bankroll=10_000.0)
    state.ticks = [_summary(e) for e in (1.0, 2.0, 3.0, 4.0, 5.0)]
    est = reward_estimate(state)
    assert est.median_usd_per_day == pytest.approx(3.0)
    assert est.median_share_of_pool == pytest.approx(0.03)
    assert est.annualized_yield_on_collateral == pytest.approx(3.0 * 365 / 1_000.0)


def test_reward_estimate_is_none_before_any_quoting():
    assert reward_estimate(MakingState()) is None


def test_phase0_gate_blocks_on_a_short_window():
    state = MakingState()
    state.ticks = [_summary(3.0)]
    gate = phase0_gate(state)
    assert not gate.ready
    assert any("ticks recorded" in b for b in gate.blockers)


def test_phase0_gate_blocks_on_an_unstable_estimate():
    state = MakingState()
    # 300 ticks spread over a long window, but wildly dispersed
    state.ticks = [_summary(e) for e in ([0.1] * 150 + [50.0] * 150)]
    state.ticks[0].timestamp = "2026-08-01T00:00:00+00:00"
    state.ticks[-1].timestamp = "2026-08-20T00:00:00+00:00"
    gate = phase0_gate(state)
    assert not gate.ready
    assert any("too unstable" in b for b in gate.blockers)


def test_universe_funnel_aggregates_across_ticks():
    state = MakingState()
    t1, t2 = _summary(1.0), _summary(1.0)
    t1.rejections = {"below_min_24h_volume": 5}
    t2.rejections = {"below_min_24h_volume": 3, "no_funded_rewards": 10}
    state.ticks = [t1, t2]
    assert universe_funnel(state) == [("no_funded_rewards", 10), ("below_min_24h_volume", 8)]
