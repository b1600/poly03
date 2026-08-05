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
