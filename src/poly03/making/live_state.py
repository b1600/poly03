"""Book M Phase 1 (strategy_v2.md §4) execution state: real inventory, real
fills, real capital -- persisted separately from `making/state.py`'s Phase 0
observation series (see MAKING_LIVE_STATE_FILE's docstring in config.py for
why the two must not share a file).

`making/execution.py`'s `reconcile_fills()`/`compute_markouts()` are what
populate this from the API each tick -- this module never assumes a fill
happened, only records one once the CLOB confirms it (v1 §5.3: "trust
reconciliation, not local assumptions").

Reward payouts are the one thing here that's manually recorded rather than
reconciled: Polymarket's liquidity rewards are paid out per epoch, not
per-order, and no endpoint in py-clob-client surfaces a per-market realized
payout to reconcile against. `record_reward_payout()` exists so an operator
can log what actually landed (from the rewards dashboard or an on-chain
transfer) against `making/rewards.py`'s Phase 0 estimate -- see §4's "check
the reconstructed scoring ... against actual payouts."
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from poly03.cluster.tagging import ClusterTags
from poly03.config import MAKING_LIVE_BANKROLL_CAP_USD, MAKING_LIVE_DECISION_LOG_FILE, MAKING_LIVE_STATE_FILE

FillSide = Literal["buy", "sell"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LiveFill:
    """One real fill, as reconciled against the API -- not an assumption.

    `mid_price_at_fill` is the midpoint our resting order was quoted
    against (`LiveOrder.quoted_midpoint`), used as the fill-time reference
    for markout rather than a fresh book fetch -- for a maker order the two
    are close in time by construction. `markout_{5,30}m_usd` are filled in
    later by `compute_markouts()` once that much time has actually passed;
    `None` means "not old enough to score yet", not zero.
    """

    id: str
    market_id: str
    condition_id: str
    token_id: str
    question: str
    side: FillSide
    price: float
    size_shares: float
    collateral_usd: float
    order_id: str
    filled_at: str = field(default_factory=_now_iso)
    mid_price_at_fill: float = 0.0
    fee_usd: float = 0.0
    markout_5m_usd: float | None = None
    markout_30m_usd: float | None = None


@dataclass
class LiveOrder:
    """One resting order we believe is live on the book. `size_matched` is
    the last size we observed filled as of the previous reconciliation --
    the execution engine diffs against it to find new fills without
    double-counting a partial fill it already recorded."""

    order_id: str
    market_id: str
    condition_id: str
    token_id: str
    question: str
    side: FillSide
    price: float
    size_shares: float
    quoted_midpoint: float
    size_matched: float = 0.0
    placed_at: str = field(default_factory=_now_iso)
    cluster_tags: dict[str, Any] = field(default_factory=dict)

    @property
    def cluster(self) -> ClusterTags:
        return _cluster_from_dict(self.market_id, self.cluster_tags)


@dataclass
class LiveInventory:
    """Net position in one market, updated as fills are recorded."""

    market_id: str
    condition_id: str
    token_id: str
    question: str
    net_shares: float = 0.0
    avg_price: float = 0.0
    realized_pnl_usd: float = 0.0
    cluster_tags: dict[str, Any] = field(default_factory=dict)

    @property
    def collateral_usd(self) -> float:
        return abs(self.net_shares) * self.avg_price

    @property
    def cluster(self) -> ClusterTags:
        return _cluster_from_dict(self.market_id, self.cluster_tags)


def _cluster_from_dict(market_id: str, tags: dict[str, Any]) -> ClusterTags:
    """§4.3 cluster caps need a ClusterTags to check against; this
    reconstructs one from the plain dict stored on the order/position (same
    shape paper/state.py's PaperPosition.cluster_tags uses). Falls back to
    treating the market as its own singleton cluster when tags haven't been
    stamped yet, rather than lumping untagged markets into one shared
    'unknown' bucket that could trip a cap for unrelated markets."""
    return ClusterTags(
        market_id=market_id,
        entity=tags.get("entity", market_id),
        themes=tuple(tags.get("themes", ())),
        geography=tags.get("geography"),
        resolution_source=tags.get("resolution_source", market_id),
        date_bucket=tags.get("date_bucket"),
    )


@dataclass
class LiveMakingState:
    """Phase 1 execution state. `bankroll_cap_usd` is the hard §4 ceiling
    (~$500), independent of and much smaller than Phase 0's `MakingState.
    bankroll`, which is a sizing denominator, not a real balance."""

    bankroll_cap_usd: float = MAKING_LIVE_BANKROLL_CAP_USD
    cash_usd: float = MAKING_LIVE_BANKROLL_CAP_USD
    positions: list[LiveInventory] = field(default_factory=list)
    open_orders: list[LiveOrder] = field(default_factory=list)
    fills: list[LiveFill] = field(default_factory=list)
    reward_payouts: list[dict[str, Any]] = field(default_factory=list)
    realized_reward_usd_total: float = 0.0
    realized_fee_usd_total: float = 0.0
    halted: bool = False
    halt_reasons: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    last_reconciled_at: str | None = None
    n_ticks: int = 0

    @property
    def open_positions(self) -> list[LiveInventory]:
        return [p for p in self.positions if p.net_shares != 0.0]

    @property
    def deployed_collateral_usd(self) -> float:
        return sum(p.collateral_usd for p in self.open_positions)

    @property
    def equity_usd(self) -> float:
        return self.cash_usd + self.deployed_collateral_usd

    def orders_for(self, market_id: str) -> list[LiveOrder]:
        return [o for o in self.open_orders if o.market_id == market_id]

    def add_order(self, order: LiveOrder) -> None:
        self.open_orders.append(order)

    def remove_order(self, order_id: str) -> None:
        self.open_orders = [o for o in self.open_orders if o.order_id != order_id]

    def position_for(self, market_id: str, *, condition_id: str, token_id: str, question: str) -> LiveInventory:
        for p in self.positions:
            if p.market_id == market_id:
                return p
        pos = LiveInventory(market_id=market_id, condition_id=condition_id, token_id=token_id, question=question)
        self.positions.append(pos)
        return pos

    def record_fill(
        self,
        *,
        market_id: str,
        condition_id: str,
        token_id: str,
        question: str,
        side: FillSide,
        price: float,
        size_shares: float,
        order_id: str,
        mid_price_at_fill: float = 0.0,
        fee_usd: float = 0.0,
    ) -> LiveFill:
        """Apply one real fill (buy adds to inventory, sell reduces it) and
        update cash/avg price/realized P&L accordingly. This is the single
        place inventory changes -- the execution engine should call this
        once per confirmed fill, never mutate `positions` directly."""
        collateral_usd = price * size_shares
        signed_shares = size_shares if side == "buy" else -size_shares

        pos = self.position_for(market_id, condition_id=condition_id, token_id=token_id, question=question)
        new_net = pos.net_shares + signed_shares

        if pos.net_shares == 0.0 or (pos.net_shares > 0) == (signed_shares > 0):
            # Adding to (or opening) a position on the same side: roll the
            # average price forward.
            total_cost = pos.avg_price * abs(pos.net_shares) + price * abs(signed_shares)
            pos.avg_price = total_cost / abs(new_net) if new_net != 0 else 0.0
        else:
            # Reducing or flipping: realize P&L on the closed portion at the
            # existing avg_price before any remaining shares get a new basis.
            closed_shares = min(abs(signed_shares), abs(pos.net_shares))
            pnl_per_share = (price - pos.avg_price) if pos.net_shares > 0 else (pos.avg_price - price)
            pos.realized_pnl_usd += pnl_per_share * closed_shares
            if abs(signed_shares) > abs(pos.net_shares):
                pos.avg_price = price  # flipped past flat, new basis on the remainder

        pos.net_shares = new_net
        self.cash_usd += (-collateral_usd if side == "buy" else collateral_usd) - fee_usd
        self.realized_fee_usd_total += fee_usd

        fill = LiveFill(
            id=str(uuid.uuid4()),
            market_id=market_id,
            condition_id=condition_id,
            token_id=token_id,
            question=question,
            side=side,
            price=price,
            size_shares=size_shares,
            collateral_usd=collateral_usd,
            order_id=order_id,
            mid_price_at_fill=mid_price_at_fill,
            fee_usd=fee_usd,
        )
        self.fills.append(fill)
        return fill

    def record_reward_payout(self, amount_usd: float, *, note: str = "") -> None:
        """Log a reward payout observed out-of-band (rewards dashboard or an
        on-chain transfer) -- see the module docstring for why this can't be
        reconciled automatically. Adds straight to cash, same as a fill
        would, since it's real USDC that landed."""
        self.reward_payouts.append({"amount_usd": amount_usd, "note": note, "recorded_at": _now_iso()})
        self.realized_reward_usd_total += amount_usd
        self.cash_usd += amount_usd


def _state_from_dict(raw: dict[str, Any]) -> LiveMakingState:
    positions = [LiveInventory(**p) for p in raw.pop("positions", [])]
    open_orders = [LiveOrder(**o) for o in raw.pop("open_orders", [])]
    fills = [LiveFill(**f) for f in raw.pop("fills", [])]
    state = LiveMakingState(**raw)
    state.positions = positions
    state.open_orders = open_orders
    state.fills = fills
    return state


def load_state(path: str | Path = MAKING_LIVE_STATE_FILE) -> LiveMakingState:
    p = Path(path)
    if not p.exists():
        return LiveMakingState()
    return _state_from_dict(json.loads(p.read_text()))


def save_state(state: LiveMakingState, path: str | Path = MAKING_LIVE_STATE_FILE) -> None:
    state.updated_at = _now_iso()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(state), indent=2, default=str))


def log_event(event: dict[str, Any], path: str | Path = MAKING_LIVE_DECISION_LOG_FILE) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"timestamp": _now_iso(), **event}, default=str) + "\n")
