"""Rebuild `making_live_state.json` from what the exchange actually holds.

The live state is the bot's *belief* about its own book, and that belief has
diverged from reality in every way that mattered:

- 2026-08-31: phantom double-booked fills produced a $58 equity gap, which
  tripped a drawdown halt at a floor the account had never really breached.
- 2026-09-01 02:46: a hand-written `making_live_state.rebuilt.json` declared
  `positions: []` and `cash: $480.65` after an incident. The wallet still
  held roughly $240 of inventory from the previous session. The book then
  quoted for six hours against caps computed from a position of zero, and
  the local ledger's idea of free cash ($390) was $273 above the exchange's
  ($117.04).

Reconstructing from an API read instead of by hand removes the failure mode
that both of those share: a number typed (or inferred) rather than observed.

Sources, and why each:

- **positions**: the public data API's `/positions`. It reports `avgPrice`
  as the exchange computed it across every fill, including ones the bot
  never saw.
- **cash**: the CLOB's USDC collateral balance. Needs L2 creds; pass
  `cash_usd` explicitly to skip it.
- **market identity**: Gamma, batched by `condition_id`. The bot keys
  positions on Gamma's numeric market id, and a position carrying anything
  else can never match the quotable universe -- it would be held forever and
  never quoted.
- **open orders**: the CLOB's own list, so a resting order placed by a
  previous process is adopted rather than duplicated.

Resolved positions are deliberately *excluded*: they are redemption claims,
not tradeable inventory, and including them would have the engine try to
flatten something with no book.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import requests

from poly03.data.clob import ClobClient
from poly03.data.gamma import GammaClient
from poly03.making.live_state import LiveInventory, LiveMakingState, LiveOrder

logger = logging.getLogger("poly03.making.rebuild")

DATA_API = "https://data-api.polymarket.com"
_GAMMA_CONDITION_BATCH = 20


@dataclass
class RebuildReport:
    """What the rebuild saw, so the operator can check it against the UI
    before trusting the state file it just wrote."""

    positions: list[LiveInventory] = field(default_factory=list)
    orders: list[LiveOrder] = field(default_factory=list)
    cash_usd: float = 0.0
    redeemable: list[dict] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    chain_mismatches: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def inventory_cost_usd(self) -> float:
        return sum(p.collateral_usd for p in self.positions)

    @property
    def inventory_value_usd(self) -> float:
        return sum(p.market_value_usd for p in self.positions)

    @property
    def redeemable_value_usd(self) -> float:
        return sum(float(p.get("currentValue") or 0.0) for p in self.redeemable)


def fetch_exchange_positions(address: str, *, timeout: float = 30.0) -> list[dict]:
    """Every open position the exchange attributes to this wallet."""
    resp = requests.get(
        f"{DATA_API}/positions",
        params={"user": address, "sizeThreshold": 0.01, "limit": 500},
        timeout=timeout,
    )
    resp.raise_for_status()
    return resp.json()


def _gamma_markets_by_condition(gamma: GammaClient, condition_ids: list[str]) -> dict[str, Any]:
    """condition_id -> Market, resolved in batches.

    Gamma's `/markets?condition_ids=` accepts repeats of the param, which is
    far cheaper than one request per position and much cheaper than scanning.
    A closed or delisted market simply doesn't come back; the caller reports
    those rather than guessing an id for them.
    """
    out: dict[str, Any] = {}
    for i in range(0, len(condition_ids), _GAMMA_CONDITION_BATCH):
        batch = condition_ids[i : i + _GAMMA_CONDITION_BATCH]
        try:
            raw = gamma._get("/markets", params={"condition_ids": batch})
        except Exception as exc:
            logger.warning("gamma condition lookup failed for %d ids: %s", len(batch), exc)
            continue
        for item in raw or []:
            try:
                market = gamma._market_from_raw(item)
            except Exception:
                continue
            if market.condition_id:
                out[market.condition_id] = market
    return out


def build_state(
    *,
    address: str,
    bankroll_cap_usd: float,
    gamma: GammaClient,
    clob: ClobClient | None = None,
    cash_usd: float | None = None,
    verify_chain: bool = False,
) -> tuple[LiveMakingState, RebuildReport]:
    """Reconstruct a `LiveMakingState` from the exchange. Read-only."""
    report = RebuildReport()
    raw_positions = fetch_exchange_positions(address)

    live_raw = []
    for p in raw_positions:
        if p.get("redeemable") or float(p.get("curPrice") or 0.0) <= 0.0:
            # A resolved (or worthless) position is a redemption claim, not
            # inventory to quote around or flatten.
            report.redeemable.append(p)
            continue
        live_raw.append(p)

    condition_ids = sorted({p["conditionId"] for p in live_raw if p.get("conditionId")})
    markets = _gamma_markets_by_condition(gamma, condition_ids) if condition_ids else {}

    for p in live_raw:
        cond = p.get("conditionId") or ""
        market = markets.get(cond)
        if market is None:
            # Keyed on condition_id so the position is still *counted* --
            # dropping it is what let deployed capital and cluster caps be
            # computed against a book that was missing $240 of inventory.
            # It just can't be matched to a quotable market, so the engine
            # will hold it and never quote it.
            report.unresolved.append(f"{p.get('title', '?')[:60]} ({cond[:12]}...)")
            market_id, tick, neg_risk, end_date = cond, 0.01, bool(p.get("negativeRisk")), p.get("endDate")
        else:
            market_id = market.id
            tick = float(market.order_price_min_tick_size or 0.01)
            neg_risk = bool(market.neg_risk)
            end_date = market.end_date.isoformat() if market.end_date else p.get("endDate")

        shares = float(p["size"])
        if verify_chain and clob is not None:
            try:
                on_chain = clob.get_conditional_balance(p["asset"])
            except Exception as exc:
                report.notes.append(f"chain balance check failed for {p.get('title', '?')[:40]}: {exc}")
            else:
                if abs(on_chain - shares) > 0.01:
                    report.chain_mismatches.append(
                        f"{p.get('title', '?')[:44]}: data-api {shares:g} vs on-chain {on_chain:g} -- using on-chain"
                    )
                    shares = on_chain

        if shares <= 0:
            continue

        report.positions.append(
            LiveInventory(
                market_id=str(market_id),
                condition_id=cond,
                token_id=p["asset"],
                question=p.get("title", ""),
                net_shares=shares,
                avg_price=float(p["avgPrice"]),
                mark_price=float(p["curPrice"]),
                tick_size=tick,
                neg_risk=neg_risk,
                end_date_iso=end_date,
            )
        )

    # Resting orders: adopt what the exchange says is live, so a restart
    # neither duplicates them nor leaves them unmanaged.
    if clob is not None:
        try:
            remote_orders = clob.get_open_orders()
        except Exception as exc:
            report.notes.append(f"could not read open orders ({exc}) -- state will start with none tracked")
            remote_orders = []
        by_token = {pos.token_id: pos for pos in report.positions}
        for o in remote_orders:
            token_id = o.get("asset_id") or o.get("token_id")
            order_id = o.get("id")
            price, size = o.get("price"), o.get("original_size", o.get("size"))
            if not (token_id and order_id and price is not None and size is not None):
                continue
            ctx = by_token.get(token_id)
            report.orders.append(
                LiveOrder(
                    order_id=str(order_id),
                    market_id=ctx.market_id if ctx else str(token_id),
                    condition_id=ctx.condition_id if ctx else "",
                    token_id=str(token_id),
                    question=ctx.question if ctx else "(order in a market with no position)",
                    # Book M only ever rests BUYs; a SELL here is a flatten
                    # order from a previous run, which is still ours to track.
                    side="sell" if str(o.get("side", "")).upper() == "SELL" else "buy",
                    price=float(price),
                    size_shares=float(size),
                    quoted_midpoint=float(price),
                    size_matched=float(o.get("size_matched") or 0.0),
                    tick_size=ctx.tick_size if ctx else 0.01,
                    neg_risk=ctx.neg_risk if ctx else False,
                    end_date_iso=ctx.end_date_iso if ctx else None,
                )
            )

    if cash_usd is None:
        if clob is None:
            raise ValueError("cash_usd must be given when no ClobClient is available")
        cash_usd, _allowance = clob.get_usdc_balance_allowance()
    report.cash_usd = float(cash_usd)

    state = LiveMakingState(
        bankroll_cap_usd=bankroll_cap_usd,
        cash_usd=report.cash_usd,
        positions=list(report.positions),
        open_orders=list(report.orders),
        # Everything before this instant is already reflected in the
        # positions above, so trade history must not be replayed onto them --
        # that is exactly the double-book the 2026-08-31 incident was.
        trades_booked_through=datetime.now(timezone.utc).timestamp(),
    )
    return state, report
