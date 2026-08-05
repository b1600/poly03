"""Typed wrappers around raw Gamma/CLOB JSON.

Gamma encodes several fields (outcomes, outcomePrices, clobTokenIds,
umaResolutionStatuses) as JSON-*strings*, not native JSON arrays. The
validators below unwrap those so nothing downstream has to know that.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from dateutil import parser as dtparser
from pydantic import BaseModel, Field, field_validator


def _parse_dt(v: Any) -> datetime | None:
    """Gamma isn't consistent about datetime formatting (e.g. closedTime
    comes back as '2026-05-12 06:44:09+00', which isn't valid ISO-8601 --
    dateutil handles it, stdlib fromisoformat doesn't)."""
    if v in (None, ""):
        return None
    if isinstance(v, datetime):
        return v
    return dtparser.parse(v)


def _parse_json_list(val: Any) -> list:
    if val is None:
        return []
    if isinstance(val, list):
        return val
    if isinstance(val, str):
        if val == "":
            return []
        return json.loads(val)
    return list(val)


class FeeSchedule(BaseModel):
    """Raw fee-schedule passthrough. Shape varies by market (e.g. sports
    markets carry rate/exponent/rebateRate; others may carry none of this).
    Do not assume zero fees anywhere downstream -- see strategy_v1.md §1.2.
    """

    model_config = {"extra": "allow"}

    rate: float | None = None
    exponent: float | None = None
    takerOnly: bool | None = None
    rebateRate: float | None = None


class Market(BaseModel):
    """One binary (Yes/No) market. Multi-outcome questions (e.g. '2028 GOP
    nominee') are modeled by Polymarket as one Event containing many of
    these, one per candidate -- see Event below.
    """

    model_config = {"extra": "ignore"}

    id: str
    condition_id: str = Field(alias="conditionId")
    question: str
    slug: str
    description: str = ""
    resolution_source: str = Field(default="", alias="resolutionSource")
    group_item_title: str = Field(default="", alias="groupItemTitle")

    outcomes: list[str] = Field(default_factory=list)
    outcome_prices: list[float] = Field(default_factory=list, alias="outcomePrices")
    clob_token_ids: list[str] = Field(default_factory=list, alias="clobTokenIds")

    start_date: datetime | None = Field(default=None, alias="startDate")
    end_date: datetime | None = Field(default=None, alias="endDate")
    closed_time: datetime | None = Field(default=None, alias="closedTime")

    active: bool = True
    closed: bool = False
    archived: bool = False
    accepting_orders: bool = Field(default=False, alias="acceptingOrders")

    volume: float = 0.0
    volume_24hr: float = Field(default=0.0, alias="volume24hr")
    liquidity: float = 0.0

    order_price_min_tick_size: float = Field(default=0.001, alias="orderPriceMinTickSize")
    order_min_size: float = Field(default=5.0, alias="orderMinSize")

    neg_risk: bool = Field(default=False, alias="negRisk")
    uma_resolution_status: str | None = Field(default=None, alias="umaResolutionStatus")
    uma_resolution_statuses: list[str] = Field(default_factory=list, alias="umaResolutionStatuses")

    best_bid: float | None = Field(default=None, alias="bestBid")
    best_ask: float | None = Field(default=None, alias="bestAsk")
    spread: float | None = None

    fee_type: str | None = Field(default=None, alias="feeType")
    fee_schedule: FeeSchedule | None = Field(default=None, alias="feeSchedule")
    maker_base_fee: float | None = Field(default=None, alias="makerBaseFee")
    taker_base_fee: float | None = Field(default=None, alias="takerBaseFee")

    # populated from the parent event by the Gamma client, not present on the raw market payload
    event_id: str | None = None
    event_title: str | None = None
    tags: list[str] = Field(default_factory=list)
    open_interest: float | None = None

    raw: dict[str, Any] = Field(default_factory=dict, exclude=True)

    @field_validator("outcomes", "clob_token_ids", "uma_resolution_statuses", mode="before")
    @classmethod
    def _v_str_list(cls, v: Any) -> list:
        return _parse_json_list(v)

    @field_validator("start_date", "end_date", "closed_time", mode="before")
    @classmethod
    def _v_dt(cls, v: Any) -> datetime | None:
        return _parse_dt(v)

    @field_validator("outcome_prices", mode="before")
    @classmethod
    def _v_prices(cls, v: Any) -> list[float]:
        return [float(x) for x in _parse_json_list(v)]

    @property
    def days_to_resolution(self) -> float | None:
        if self.end_date is None:
            return None
        now = datetime.now(timezone.utc)
        return (self.end_date - now).total_seconds() / 86400.0

    @property
    def is_resolved(self) -> bool:
        return self.closed and self.uma_resolution_status == "resolved"

    @property
    def winning_outcome_index(self) -> int | None:
        """Index into `outcomes`/`outcome_prices` of the settled winner, or
        None if not (yet) unambiguously resolved to 0/1."""
        if not self.is_resolved or len(self.outcome_prices) != 2:
            return None
        if self.outcome_prices[0] == 1.0:
            return 0
        if self.outcome_prices[1] == 1.0:
            return 1
        return None

    def token_id_for_outcome(self, outcome: str) -> str | None:
        try:
            idx = self.outcomes.index(outcome)
        except ValueError:
            return None
        return self.clob_token_ids[idx] if idx < len(self.clob_token_ids) else None


class Event(BaseModel):
    """A group of one or more binary Markets. Multi-outcome political
    questions live here as N markets, one per candidate/outcome.
    """

    model_config = {"extra": "ignore"}

    id: str
    slug: str
    title: str
    resolution_source: str = Field(default="", alias="resolutionSource")
    end_date: datetime | None = Field(default=None, alias="endDate")
    closed: bool = False
    volume: float = 0.0
    open_interest: float = Field(default=0.0, alias="openInterest")
    neg_risk: bool = Field(default=False, alias="enableNegRisk")
    tags: list[str] = Field(default_factory=list)
    market_ids: list[str] = Field(default_factory=list)
    raw_markets: list[dict[str, Any]] = Field(default_factory=list, exclude=True)

    @field_validator("end_date", mode="before")
    @classmethod
    def _v_dt(cls, v: Any) -> datetime | None:
        return _parse_dt(v)

    @field_validator("tags", mode="before")
    @classmethod
    def _v_tags(cls, v: Any) -> list[str]:
        if not v:
            return []
        out = []
        for t in v:
            if isinstance(t, str):
                out.append(t)
            elif isinstance(t, dict) and "label" in t:
                out.append(t["label"])
        return out

    @property
    def is_multi_outcome(self) -> bool:
        return len(self.market_ids) > 1


class PriceLevel(BaseModel):
    price: float
    size: float

    @field_validator("price", "size", mode="before")
    @classmethod
    def _v_float(cls, v: Any) -> float:
        return float(v)


class OrderBook(BaseModel):
    market: str = ""
    asset_id: str = Field(default="", alias="asset_id")
    bids: list[PriceLevel] = Field(default_factory=list)
    asks: list[PriceLevel] = Field(default_factory=list)
    timestamp: datetime | None = None

    model_config = {"extra": "ignore"}

    @field_validator("timestamp", mode="before")
    @classmethod
    def _v_ts(cls, v: Any) -> datetime | None:
        if v in (None, ""):
            return None
        return datetime.fromtimestamp(int(v) / 1000.0, tz=timezone.utc)

    @property
    def best_bid(self) -> PriceLevel | None:
        return max(self.bids, key=lambda l: l.price) if self.bids else None

    @property
    def best_ask(self) -> PriceLevel | None:
        return min(self.asks, key=lambda l: l.price) if self.asks else None

    @property
    def spread(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return ba.price - bb.price

    def depth_within(self, side: str, price_limit: float) -> float:
        """Total size available at price <= price_limit (asks) or
        price >= price_limit (bids)."""
        levels = self.asks if side == "ask" else self.bids
        if side == "ask":
            return sum(l.size for l in levels if l.price <= price_limit)
        return sum(l.size for l in levels if l.price >= price_limit)
