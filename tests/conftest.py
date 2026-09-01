from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

import pytest

from poly03.data.models import Market


def make_market(
    question: str,
    description: str = "",
    *,
    days_to_resolution: float | None = 100,
    resolution_source: str = "",
    closed: bool = False,
    outcome_prices: tuple[float, float] = (0.5, 0.5),
    volume_24hr: float = 5_000.0,
    open_interest: float | None = 50_000.0,
    best_bid: float | None = None,
    best_ask: float | None = None,
    group_item_title: str = "",
    tags: list[str] | None = None,
    accepting_orders: bool = True,
    uma_resolution_status: str | None = None,
) -> Market:
    end_date = None
    if days_to_resolution is not None:
        end_date = datetime.now(timezone.utc) + timedelta(days=days_to_resolution)

    m = Market(
        id="test-1",
        conditionId="0xabc",
        question=question,
        slug="test-market",
        description=description,
        resolutionSource=resolution_source,
        groupItemTitle=group_item_title,
        outcomes=json.dumps(["Yes", "No"]),
        outcomePrices=json.dumps([str(p) for p in outcome_prices]),
        clobTokenIds=json.dumps(["111", "222"]),
        endDate=end_date.isoformat() if end_date else None,
        closed=closed,
        acceptingOrders=accepting_orders,
        volume24hr=volume_24hr,
        bestBid=best_bid,
        bestAsk=best_ask,
        umaResolutionStatus=uma_resolution_status,
    )
    m.open_interest = open_interest
    m.tags = tags or []
    return m


@pytest.fixture
def market_factory():
    return make_market


@pytest.fixture(autouse=True)
def _never_write_to_live_run_state(monkeypatch, tmp_path):
    """Keep the test suite out of the real run-state files.

    Several execution tests call `run_live_tick`/`_place_side` without
    passing `decision_log_path`, so they fell through to the module default
    -- `making_live_decisions.jsonl`, the *production* decision log that
    `make live report` reads and that this repo tracks in git. A test run
    appended ~80 fabricated `market-a`/`order-1` entries to it, which then
    showed up in analysis as real placements and cancels made minutes after
    the live loop had actually stopped.

    Rebinding the module constants is not enough: `run_live_tick`'s
    `decision_log_path` and `log_event`'s `path` take the constant as a
    *default argument*, which Python evaluates once at def time, so a later
    monkeypatch of the module attribute is never consulted. The write itself
    is what has to be intercepted -- so `log_event` is wrapped, and every
    call is forced into tmp_path regardless of the path it was given."""
    import poly03.making.execution as execution
    import poly03.making.live_state as live_state

    log = tmp_path / "test_decisions.jsonl"
    real_log_event = live_state.log_event

    def _sandboxed_log_event(event, path=None):  # noqa: ARG001 - path deliberately ignored
        return real_log_event(event, path=log)

    monkeypatch.setattr(live_state, "log_event", _sandboxed_log_event)
    monkeypatch.setattr(execution, "log_event", _sandboxed_log_event)
    monkeypatch.setattr(live_state, "MAKING_LIVE_STATE_FILE", str(tmp_path / "state.json"), raising=False)
