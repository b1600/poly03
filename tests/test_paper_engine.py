from __future__ import annotations

from poly03.classifier.taxonomy import Tier
from poly03.paper.engine import check_kill_switches, run_tick, scan_universe
from poly03.paper.state import PaperState


class FakeGamma:
    """Minimal stand-in for GammaClient. Markets are looked up by id so
    tests can mutate `self.markets[id]` between ticks to simulate the world
    changing (resolution, price moves, reclassification)."""

    def __init__(self, markets):
        self.markets = {m.id: m for m in markets}

    def iter_markets(self, **kwargs):
        yield from list(self.markets.values())

    def get_market(self, market_id):
        return self.markets[market_id]

    def get_event(self, event_id):
        raise AssertionError("get_event should not be called when market.event_id is unset in tests")


def _estimator(q: float = 0.95):
    """Stand-in for the edge estimator Book A now requires (strategy_v2.md
    §1.1). Tests that exercise entry have to supply one explicitly -- that is
    the point of the guard: without it, nothing enters."""
    return lambda market: q


def _tier1_market(market_factory, *, market_id="m1", best_bid=0.90, best_ask=0.91, days=100.0):
    m = market_factory(
        "Will the referendum be held on schedule?",
        description="The referendum has not been scheduled yet by the official election commission.",
        resolution_source="Official election commission release",
        best_bid=best_bid,
        best_ask=best_ask,
        days_to_resolution=days,
    )
    m.id = market_id
    m.liquidity = 50_000.0
    return m


def test_scan_universe_finds_tier1_candidate(market_factory, tmp_path):
    gamma = FakeGamma([_tier1_market(market_factory)])
    from poly03.classifier.llm_veto import NoOpVeto

    candidates, _scanned = scan_universe(
        gamma,
        veto=NoOpVeto(),
        max_markets=10,
        target_size_usd=100.0,
        exclude_market_ids=set(),
        log_path=str(tmp_path / "log.jsonl"),
        edge_estimator=_estimator(),
    )
    assert len(candidates) == 1
    assert candidates[0].classification.tier == Tier.TIER_1


def test_scan_universe_rejects_out_of_band_price(market_factory, tmp_path):
    gamma = FakeGamma([_tier1_market(market_factory, best_bid=0.50, best_ask=0.51)])
    from poly03.classifier.llm_veto import NoOpVeto

    candidates, _scanned = scan_universe(
        gamma,
        veto=NoOpVeto(),
        max_markets=10,
        target_size_usd=100.0,
        exclude_market_ids=set(),
        log_path=str(tmp_path / "log.jsonl"),
        edge_estimator=_estimator(),
    )
    assert candidates == []


def test_scan_universe_rejects_ambiguous_resolution(market_factory, tmp_path):
    m = market_factory(
        "Will the referendum be held on schedule?",
        description="This will resolve based on what is widely reported as the outcome.",
        resolution_source="Official election commission release",
        best_bid=0.90,
        best_ask=0.91,
    )
    m.id = "m2"
    m.liquidity = 50_000.0
    gamma = FakeGamma([m])
    from poly03.classifier.llm_veto import NoOpVeto

    log_path = tmp_path / "log.jsonl"
    candidates, _scanned = scan_universe(
        gamma,
        veto=NoOpVeto(),
        max_markets=10,
        target_size_usd=100.0,
        exclude_market_ids=set(),
        log_path=str(log_path),
        edge_estimator=_estimator(),
    )
    assert candidates == []
    logged = log_path.read_text()
    assert "ambiguous_resolution_criteria" in logged


def test_run_tick_enters_a_position(market_factory, tmp_path):
    gamma = FakeGamma([_tier1_market(market_factory)])
    state = PaperState(bankroll=100_000.0, cash=100_000.0, high_water_mark=100_000.0)

    report = run_tick(
        state,
        gamma=gamma,
        clob=object(),
        decision_log_path=str(tmp_path / "log.jsonl"),
        edge_estimator=_estimator(),
    )

    assert len(report.entered) == 1
    assert len(state.open_positions) == 1
    pos = state.open_positions[0]
    assert pos.tier == 1
    assert pos.entry_price == 0.90
    assert state.cash < 100_000.0


def test_run_tick_resolves_winning_position(market_factory, tmp_path):
    market = _tier1_market(market_factory)
    gamma = FakeGamma([market])
    state = PaperState(bankroll=100_000.0, cash=100_000.0, high_water_mark=100_000.0)
    log_path = str(tmp_path / "log.jsonl")

    run_tick(state, gamma=gamma, clob=object(), decision_log_path=log_path, edge_estimator=_estimator())
    assert len(state.open_positions) == 1

    # simulate the world advancing: market resolves Yes (outcome index 0 wins)
    resolved = market_factory(
        market.question,
        description=market.description,
        resolution_source=market.resolution_source,
        closed=True,
        outcome_prices=(1.0, 0.0),
        uma_resolution_status="resolved",
    )
    resolved.id = market.id
    gamma.markets[market.id] = resolved

    report = run_tick(state, gamma=gamma, clob=object(), decision_log_path=log_path, edge_estimator=_estimator())
    assert report.resolved_win == 1
    assert state.open_positions == []
    win = state.closed_positions[0]
    assert win.status == "resolved_win"
    assert win.realized_pnl > 0


def test_run_tick_exits_on_adverse_price_move(market_factory, tmp_path):
    market = _tier1_market(market_factory)
    gamma = FakeGamma([market])
    state = PaperState(bankroll=100_000.0, cash=100_000.0, high_water_mark=100_000.0)
    log_path = str(tmp_path / "log.jsonl")

    run_tick(state, gamma=gamma, clob=object(), decision_log_path=log_path, edge_estimator=_estimator())
    assert len(state.open_positions) == 1

    # price craters well past the 8c adverse-move threshold
    dropped = _tier1_market(market_factory, best_bid=0.70, best_ask=0.71)
    dropped.id = market.id
    gamma.markets[market.id] = dropped

    report = run_tick(state, gamma=gamma, clob=object(), decision_log_path=log_path, edge_estimator=_estimator())
    assert ("adverse_price_move" in [r for _, r in report.exited])
    assert state.open_positions == []
    exited = state.closed_positions[0]
    assert exited.status == "exited_early"
    assert exited.close_reason == "adverse_price_move"


def test_run_tick_exits_on_tier_downgrade(market_factory, tmp_path):
    market = _tier1_market(market_factory)
    gamma = FakeGamma([market])
    state = PaperState(bankroll=100_000.0, cash=100_000.0, high_water_mark=100_000.0)
    log_path = str(tmp_path / "log.jsonl")

    run_tick(state, gamma=gamma, clob=object(), decision_log_path=log_path, edge_estimator=_estimator())
    assert len(state.open_positions) == 1

    # rewrite the market so the rules engine no longer sees a tier-1 signature
    downgraded = market_factory(
        "Will something unrelated and unforecastable happen?",
        description="No structural signal here at all.",
        resolution_source="Official election commission release",
        best_bid=0.90,
        best_ask=0.91,
    )
    downgraded.id = market.id
    gamma.markets[market.id] = downgraded

    report = run_tick(state, gamma=gamma, clob=object(), decision_log_path=log_path, edge_estimator=_estimator())
    assert ("tier_downgrade" in [r for _, r in report.exited])
    exited = state.closed_positions[0]
    assert exited.close_reason == "tier_downgrade"


def test_run_tick_exits_on_dispute(market_factory, tmp_path):
    market = _tier1_market(market_factory)
    gamma = FakeGamma([market])
    state = PaperState(bankroll=100_000.0, cash=100_000.0, high_water_mark=100_000.0)
    log_path = str(tmp_path / "log.jsonl")

    run_tick(state, gamma=gamma, clob=object(), decision_log_path=log_path, edge_estimator=_estimator())

    disputed = _tier1_market(market_factory)
    disputed.id = market.id
    disputed.uma_resolution_statuses = ["disputed"]
    gamma.markets[market.id] = disputed

    report = run_tick(state, gamma=gamma, clob=object(), decision_log_path=log_path, edge_estimator=_estimator())
    assert ("dispute_filed" in [r for _, r in report.exited])


def test_check_kill_switches_tier1_miss_requires_manual_review():
    state = PaperState(bankroll=1_000.0, cash=1_000.0, high_water_mark=1_000.0)
    from poly03.cluster.tagging import ClusterTags

    pos = state.new_position(
        market_id="m1",
        question="q",
        token_id="t1",
        outcome="Yes",
        side_index=0,
        tier=1,
        entry_price=0.95,
        stake_usd=50.0,
        end_date=None,
        days_to_resolution_at_entry=30.0,
        modeled_annualized_roc=0.1,
        cluster_tags=ClusterTags(
            market_id="m1", entity="E", themes=(), geography=None, resolution_source="x", date_bucket=None
        ),
    )
    state.close_position(pos, status="resolved_loss", reason="resolution", close_price=0.0)

    halted, reasons, manual_review = check_kill_switches(state)
    assert halted is True
    assert manual_review is True
    assert any("Tier 1" in r for r in reasons)


def test_check_kill_switches_drawdown():
    state = PaperState(bankroll=1_000.0, cash=1_000.0, high_water_mark=1_000.0)
    state.cash = 800.0  # equity now 800, hwm 1000 -> 20% drawdown > 15% kill threshold
    halted, reasons, manual_review = check_kill_switches(state)
    assert halted is True
    assert any("drawdown" in r for r in reasons)
    assert manual_review is False


def test_max_concurrent_positions_throttles_entries(market_factory, tmp_path, monkeypatch):
    import poly03.paper.engine as engine_mod

    monkeypatch.setattr(engine_mod, "MAX_CONCURRENT_POSITIONS", 1)
    monkeypatch.setattr(engine_mod, "PAPER_MAX_NEW_POSITIONS_PER_TICK", 5)

    m1 = _tier1_market(market_factory, market_id="m1")
    m2 = _tier1_market(market_factory, market_id="m2")
    gamma = FakeGamma([m1, m2])
    state = PaperState(bankroll=100_000.0, cash=100_000.0, high_water_mark=100_000.0)

    report = run_tick(state, gamma=gamma, clob=object(), decision_log_path=str(tmp_path / "log.jsonl"), edge_estimator=_estimator())
    assert len(report.entered) == 1
    assert len(state.open_positions) == 1


def test_run_tick_refuses_to_enter_without_an_edge_estimate(market_factory, tmp_path):
    """strategy_v2.md §1.1: with no estimator, Book A must stay flat rather
    than trade against the placeholder q that made the margin gate a
    tautology. The rejection has to be logged, not silent."""
    gamma = FakeGamma([_tier1_market(market_factory)])
    state = PaperState(bankroll=100_000.0, cash=100_000.0, high_water_mark=100_000.0)
    log_path = tmp_path / "log.jsonl"

    report = run_tick(state, gamma=gamma, clob=object(), decision_log_path=str(log_path))

    assert report.entered == []
    assert state.open_positions == []
    assert "no_edge_estimate_available" in log_path.read_text()


def test_run_tick_reports_actual_scan_count_not_the_cap(market_factory, tmp_path):
    """strategy_v2.md §5.4: report.scanned used to echo max_markets whether or
    not Gamma returned that many."""
    gamma = FakeGamma([_tier1_market(market_factory, market_id="m1"), _tier1_market(market_factory, market_id="m2")])
    state = PaperState(bankroll=100_000.0, cash=100_000.0, high_water_mark=100_000.0)

    report = run_tick(
        state,
        gamma=gamma,
        clob=object(),
        max_markets=300,
        decision_log_path=str(tmp_path / "log.jsonl"),
        edge_estimator=_estimator(),
    )

    assert report.scanned == 2
