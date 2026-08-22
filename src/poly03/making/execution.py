"""Book M Phase 1 (strategy_v2.md §4) execution engine.

Reuses Phase 0's universe selection and quote construction unchanged
(`making/universe.py`, `making/quoting.py`) -- the thesis and the filters
don't change between "would we quote this" and "do we quote this". What's
new here is the only code in the repo that spends real capital: placing,
cancelling, and reconciling actual resting orders against
`making/live_state.py`.

Safety model:

- `dry_run=True` is the default everywhere. In dry-run, the engine computes
  and reports exactly what it *would* place/cancel/flatten without calling
  any ClobClient write method and without mutating `state.open_orders` or
  `state.positions` -- those only change from a real, reconciled fill.
- `ClobClient`'s write methods themselves refuse to run without L2 creds
  (see data/clob.py `_require_l2`) -- this module doesn't duplicate that
  check, it just lets it raise.
- Deployed collateral is hard-capped at `state.bankroll_cap_usd`
  (MAKING_LIVE_BANKROLL_CAP_USD, ~$500), independent of and far below
  Phase 0's MAKING_MAX_DEPLOYED_FRACTION math.
- §4.3 cluster caps apply to *real* exposure (open positions + resting
  orders), not simulated ticks -- a market that would push any entity/
  theme/date-bucket/resolution-source cluster over its cap is skipped
  before it's ever quoted, same caps as Phase 0, seeded fresh each tick
  from current inventory.
- An adverse-selection kill switch halts new quoting (but not
  reconciliation or flattening) once the last N fills have all marked out
  against us -- see MAKING_LIVE_KILL_MARKOUT_* in config.py. This book has
  no `q`, so v1's Tier-1-miss/drawdown switches don't apply; this is the
  Book M equivalent.
- A market that drops out of the Phase 0 universe (including via the
  flatten-before-resolution window in `select_universe`) is unwound: any
  resting orders are cancelled and any inventory is closed out at the
  touch, never left to ride into resolution.
- Every real network call is caught per-item so one failed order/cancel
  doesn't take down the rest of the tick -- same pattern as
  `making/engine.py`'s order-book batch fetch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from poly03.classifier.llm_veto import LLMClassifierVeto, NoOpVeto
from poly03.cluster.tagging import ClusterExposureTracker, ensure_event_tags, tag_market
from poly03.config import (
    GAMMA_MAX_SCAN_MARKETS,
    MAKING_LIVE_DECISION_LOG_FILE,
    MAKING_LIVE_KILL_DRAWDOWN_FRACTION,
    MAKING_LIVE_KILL_MARKOUT_CENTS_PER_SHARE,
    MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE,
    MAKING_LIVE_MARKOUT_WINDOW_SLACK_MINUTES,
    MAKING_LIVE_MAX_DATE_BUCKET_FRACTION,
    MAKING_LIVE_MAX_ENTITY_CLUSTER_FRACTION,
    MAKING_LIVE_MAX_INVENTORY_PER_MARKET_FRACTION,
    MAKING_LIVE_MAX_RESOLUTION_SOURCE_FRACTION,
    MAKING_LIVE_MAX_THEME_CLUSTER_FRACTION,
    MAKING_LIVE_MIN_NOTIONAL_USD,
    MAKING_MAX_MARKETS_QUOTED,
    MAKING_REQUOTE_MID_MOVE_CENTS,
)
from poly03.data.clob import ClobClient
from poly03.data.gamma import GammaClient
from poly03.making.engine import _fetch_books, _inventory_cap_shares
from poly03.making.live_state import LiveInventory, LiveMakingState, LiveOrder, log_event
from poly03.making.quoting import Quote, QuotePair, build_quote_pair, needs_requote, round_to_tick
from poly03.making.universe import QuotableMarket, UniverseReport, select_universe

try:
    from py_clob_client.order_builder.constants import BUY, SELL
except ImportError:  # pragma: no cover - py-clob-client always installs this
    BUY, SELL = "BUY", "SELL"

logger = logging.getLogger("poly03.making.execution")

# Polymarket has no naked shorting: starting flat, a SELL is always rejected.
# Both legs of a resting Book M quote are therefore literal BUY orders -- a
# "bid" is a YES buy, an "ask" is economically a NO buy (see _place_side,
# which no longer uses this map -- it's always BUY there). This map is used
# only by _flatten_market, which sells inventory we actually hold (legal --
# it's not a naked short) and so genuinely needs both sides of the exchange
# API. Working assumption, not yet confirmed against the rewards docs or a
# real payout: that a resting NO bid scores the same as a YES ask under
# §2.2's formula.
_SIDE_TO_ORDER_SIDE = {"bid": BUY, "ask": SELL}
_CLOSED_STATUSES = {"MATCHED", "CANCELED", "CANCELLED", "EXPIRED", "FAILED"}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class LiveTickReport:
    timestamp: str
    dry_run: bool
    universe: UniverseReport
    would_place: list[dict] = field(default_factory=list)
    placed: list[dict] = field(default_factory=list)
    would_cancel: list[str] = field(default_factory=list)
    cancelled: list[str] = field(default_factory=list)
    flattened: list[dict] = field(default_factory=list)
    new_fills: list[dict] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)

    def skip(self, reason: str) -> None:
        self.skipped[reason] = self.skipped.get(reason, 0) + 1

    def error(self, msg: str) -> None:
        logger.warning(msg)
        self.errors.append(msg)


def _extract_order_id(resp: dict) -> str | None:
    """py-clob-client's post_order response shape has drifted across
    versions; check the common spots rather than assume one."""
    for key in ("orderID", "orderId", "id"):
        if isinstance(resp.get(key), str):
            return resp[key]
    order = resp.get("order")
    if isinstance(order, dict):
        return order.get("id") or order.get("orderID")
    return None


def _remote_size_matched(order: dict) -> float:
    for key in ("size_matched", "sizeMatched", "matched_size"):
        if key in order:
            try:
                return float(order[key])
            except (TypeError, ValueError):
                pass
    return 0.0


def _remote_status(order: dict) -> str:
    return str(order.get("status", "")).upper()


def _fee_usd(clob: ClobClient, token_id: str, price: float, size_shares: float, cache: dict[str, int | None]) -> float:
    """Best-effort realized fee for one fill. Book M only ever rests GTC
    limit orders, so under §2.1's taker-only fee schedule this should be
    ~0 (plus an unmodeled maker rebate) -- but that's the assumption Phase 1
    exists to check, not something to assume here. Whatever
    get_fee_rate_bps reports is recorded as-is; 0 if the endpoint doesn't
    answer, not a guess."""
    if token_id not in cache:
        try:
            cache[token_id] = clob.get_fee_rate_bps(token_id)
        except Exception:
            cache[token_id] = None
    bps = cache[token_id]
    if not bps:
        return 0.0
    return (bps / 10_000.0) * price * size_shares


def _adopt_untracked_order(remote: dict, token_to_market: dict[str, QuotableMarket]) -> LiveOrder | None:
    """py-clob-client's OrderArgs/create_order (checked directly against the
    installed package) expose no client-settable salt/dedup key, so there's
    no SDK-level idempotency guard against a lost `post_limit_order` response
    causing a duplicate resting order next tick. The mitigation available
    without that hook: reconstruct the `LiveOrder` from what the exchange
    actually reports and bring it under local management -- the normal
    existing/stale requote logic then sees it and won't place a second one.
    Best-effort field extraction; returns None (falls back to a warn-only
    drift signal) if the remote payload or the token can't be resolved back
    to a currently-quotable market."""
    token_id = remote.get("asset_id") or remote.get("token_id")
    order_id = remote.get("id")
    price = remote.get("price")
    size = remote.get("original_size", remote.get("size"))
    if not (token_id and order_id and price is not None and size is not None):
        return None
    qm = token_to_market.get(token_id)
    if qm is None:
        return None
    try:
        price = float(price)
        size = float(size)
    except (TypeError, ValueError):
        return None
    return LiveOrder(
        order_id=order_id,
        market_id=qm.market.id,
        condition_id=qm.market.condition_id,
        token_id=token_id,
        question=qm.market.question,
        side="buy",  # Book M only ever places BUY orders -- see _SIDE_TO_ORDER_SIDE's docstring
        price=price,
        size_shares=size,
        quoted_midpoint=qm.midpoint,
        size_matched=_remote_size_matched(remote),
        tick_size=qm.tick_size,
        neg_risk=qm.market.neg_risk,
    )


def _fetch_mid(clob: ClobClient, token_id: str, cache: dict[str, float | None]) -> float | None:
    """Current midpoint for one token, memoized per call site. Shared by
    reconciliation's fresh-fill mid capture and compute_markouts' later
    comparison mid -- both want "the book right now", just at different
    points in a fill's life."""
    if token_id not in cache:
        try:
            book = clob.get_order_book(token_id)
            bb, ba = book.best_bid, book.best_ask
            cache[token_id] = (bb.price + ba.price) / 2.0 if bb and ba else None
        except Exception:
            cache[token_id] = None
    return cache[token_id]


def reconcile_fills(
    state: LiveMakingState,
    clob: ClobClient,
    report: LiveTickReport,
    *,
    decision_log_path: str,
    token_to_market: dict[str, QuotableMarket] | None = None,
) -> None:
    """Trust the API, not local assumptions (v1 §5.3): pull every open order
    we actually have resting, in one bulk call, rather than trusting that
    `state.open_orders` is still accurate. Diff each tracked order against
    that authoritative list to turn new matched size into a recorded fill,
    drop orders that are no longer open, and adopt any order the exchange
    shows that we don't have tracked when it resolves to a currently-
    quotable market (see `_adopt_untracked_order`) -- otherwise it's still
    logged as a drift signal (a fill our engine didn't see, a manual action,
    a bug)."""
    try:
        remote_orders = clob.get_open_orders()
    except Exception as exc:
        report.error(f"reconcile: get_open_orders failed, falling back to per-order lookups: {exc}")
        remote_orders = None

    fee_cache: dict[str, int | None] = {}
    mid_cache: dict[str, float | None] = {}

    if remote_orders is not None:
        remote_by_id = {r.get("id"): r for r in remote_orders if r.get("id")}
        tracked_ids = {o.order_id for o in state.open_orders}
        untracked = set(remote_by_id) - tracked_ids
        for order_id in untracked:
            adopted = _adopt_untracked_order(remote_by_id[order_id], token_to_market or {})
            if adopted is not None:
                state.add_order(adopted)
                report.error(
                    f"reconcile: adopted untracked order {order_id} (open on exchange, not tracked locally -- "
                    "likely a lost post_limit_order response)"
                )
            else:
                report.error(f"reconcile: order {order_id} is open on the exchange but not tracked locally")

        for order in list(state.open_orders):
            remote = remote_by_id.get(order.order_id)
            if remote is None:
                # Not in the open list anymore -- get its final state once
                # to capture any last fill before we stop tracking it.
                try:
                    remote = clob.get_order(order.order_id)
                except Exception as exc:
                    report.error(f"reconcile failed for order {order.order_id}: {exc}")
                    continue
                if remote is None:
                    state.remove_order(order.order_id)
                    continue
            _apply_order_update(state, order, remote, report, fee_cache, mid_cache, clob, decision_log_path)
    else:
        for order in list(state.open_orders):
            try:
                remote = clob.get_order(order.order_id)
            except Exception as exc:
                report.error(f"reconcile failed for order {order.order_id}: {exc}")
                continue
            if remote is None:
                state.remove_order(order.order_id)
                continue
            _apply_order_update(state, order, remote, report, fee_cache, mid_cache, clob, decision_log_path)


def _apply_order_update(
    state: LiveMakingState,
    order: LiveOrder,
    remote: dict,
    report: LiveTickReport,
    fee_cache: dict[str, int | None],
    mid_cache: dict[str, float | None],
    clob: ClobClient,
    decision_log_path: str,
) -> None:
    matched = _remote_size_matched(remote)
    delta = matched - order.size_matched
    if delta > 1e-9:
        fee = _fee_usd(clob, order.token_id, order.price, delta, fee_cache)
        # task item 4: the markout baseline used to be order.quoted_midpoint
        # -- the midpoint when the order was *placed*, which can predate the
        # actual fill by however long the order rested (up to the old 30min
        # loop interval). Capture a mid at reconciliation time instead, which
        # (especially under item 3's fast cadence) is close to the fill
        # moment; fall back to quoted_midpoint only if the book fetch fails.
        fresh_mid = _fetch_mid(clob, order.token_id, mid_cache)
        fill = state.record_fill(
            market_id=order.market_id,
            condition_id=order.condition_id,
            token_id=order.token_id,
            question=order.question,
            side=order.side,
            price=order.price,
            size_shares=delta,
            order_id=order.order_id,
            mid_price_at_fill=fresh_mid if fresh_mid is not None else order.quoted_midpoint,
            fee_usd=fee,
            tick_size=order.tick_size,
            neg_risk=order.neg_risk,
        )
        report.new_fills.append(fill.__dict__)
        log_event({"kind": "fill", **fill.__dict__}, path=decision_log_path)
        order.size_matched = matched

    if _remote_status(remote) in _CLOSED_STATUSES or matched >= order.size_shares - 1e-9:
        state.remove_order(order.order_id)


_MARKOUT_HORIZONS_MINUTES = (("markout_5m_usd", 5.0), ("markout_30m_usd", 30.0))


def compute_markouts(state: LiveMakingState, clob: ClobClient, report: LiveTickReport) -> None:
    """§3.4 adverse selection, tracked directly: for each fill whose age
    falls inside `[horizon, horizon + MAKING_LIVE_MARKOUT_WINDOW_SLACK_MINUTES]`
    for a not-yet-scored horizon, fetch the market's current midpoint and
    mark the fill out against it. Signed so positive = price moved in our
    favor after the fill (we bought and it went up, or sold and it went
    down) and negative = adverse selection.

    task item 4: a fill scanned only every 30 minutes (the old default loop
    interval) would get its "5m markout" stamped 25-55 minutes late using
    whatever the mid happened to be at scan time -- silently a 30-60m
    markout mislabeled as 5m. The upper bound on the window means a horizon
    that's missed just stays `None` permanently (this function keeps
    checking every tick, but `age_minutes` only grows, so the check never
    passes again) rather than stamping a stale reading."""
    now = datetime.now(timezone.utc)
    mid_cache: dict[str, float | None] = {}

    for fill in state.fills:
        if fill.markout_5m_usd is not None and fill.markout_30m_usd is not None:
            continue
        age_minutes = (now - datetime.fromisoformat(fill.filled_at)).total_seconds() / 60.0
        sign = 1.0 if fill.side == "buy" else -1.0
        for field_name, horizon in _MARKOUT_HORIZONS_MINUTES:
            if getattr(fill, field_name) is not None:
                continue
            if age_minutes < horizon or age_minutes > horizon + MAKING_LIVE_MARKOUT_WINDOW_SLACK_MINUTES:
                continue  # not due yet, or the window was missed -- leave None
            mid = _fetch_mid(clob, fill.token_id, mid_cache)
            if mid is None:
                continue
            setattr(fill, field_name, sign * (mid - fill.price) * fill.size_shares)


def check_adverse_selection_kill_switch(state: LiveMakingState, report: LiveTickReport) -> None:
    """§4 rollout item 3: halt new quoting if the last N fills all marked out
    against us beyond the per-share threshold. Deliberately "all of the last
    N", not an average -- an average lets a run of bad fills hide inside a
    history of good ones; a maker book that just started getting picked off
    needs to stop *now*, not once it's dragged the average down. Only trips
    on fills old enough to have a 5m markout; does nothing to already-placed
    orders or open inventory -- flattening those is unwind_market's job."""
    if state.halted:
        return
    scored = [f for f in state.fills if f.markout_5m_usd is not None]
    if len(scored) < MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE:
        return
    recent = scored[-MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE:]
    threshold_usd_per_share = -MAKING_LIVE_KILL_MARKOUT_CENTS_PER_SHARE / 100.0
    per_share = [f.markout_5m_usd / f.size_shares for f in recent if f.size_shares > 0]
    if len(per_share) == len(recent) and all(m <= threshold_usd_per_share for m in per_share):
        reason = (
            f"adverse selection kill switch: last {len(recent)} scored fills all marked out worse than "
            f"{MAKING_LIVE_KILL_MARKOUT_CENTS_PER_SHARE:.1f}c/share at 5m"
        )
        state.halted = True
        state.halt_reasons.append(reason)
        report.error(reason)


def check_drawdown_kill_switch(state: LiveMakingState, report: LiveTickReport) -> None:
    """task item 5: an absolute equity-drawdown halt, separate from the
    markout-based switch above -- that one needs
    MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE scored fills (which each need a 5m
    markout) before it can trip at all, which at a small live bankroll could
    be days of exposure. This one works from fill #1: any tick where
    `equity_usd` has dropped more than MAKING_LIVE_KILL_DRAWDOWN_FRACTION
    below `bankroll_cap_usd` halts immediately."""
    if state.halted:
        return
    floor = state.bankroll_cap_usd * (1.0 - MAKING_LIVE_KILL_DRAWDOWN_FRACTION)
    if state.equity_usd <= floor:
        reason = (
            f"drawdown kill switch: equity ${state.equity_usd:,.2f} <= "
            f"{MAKING_LIVE_KILL_DRAWDOWN_FRACTION:.0%} drawdown floor ${floor:,.2f} "
            f"(bankroll cap ${state.bankroll_cap_usd:,.2f})"
        )
        state.halted = True
        state.halt_reasons.append(reason)
        report.error(reason)


def _cancel(
    state: LiveMakingState,
    clob: ClobClient,
    orders: list[LiveOrder],
    report: LiveTickReport,
    *,
    dry_run: bool,
    decision_log_path: str,
) -> None:
    if not orders:
        return
    ids = [o.order_id for o in orders]
    if dry_run:
        report.would_cancel.extend(ids)
        return
    try:
        clob.cancel_orders(ids)
    except Exception as exc:
        report.error(f"cancel failed for {ids}: {exc}")
        return
    for order_id in ids:
        state.remove_order(order_id)
    report.cancelled.extend(ids)
    log_event({"kind": "cancel", "order_ids": ids}, path=decision_log_path)


def cancel_all_tracked(
    state: LiveMakingState, clob: ClobClient, *, dry_run: bool, decision_log_path: str
) -> LiveTickReport:
    """task item 5: cancel every order we're tracking, regardless of market
    -- used on shutdown (Ctrl+C) and when halted, so resting orders don't
    sit unattended. Cancel-tracked, not the blunter `clob.cancel_all()`, so
    it never touches orders on the account that aren't Book M's."""
    report = LiveTickReport(timestamp=_now_iso(), dry_run=dry_run, universe=UniverseReport())
    _cancel(state, clob, list(state.open_orders), report, dry_run=dry_run, decision_log_path=decision_log_path)
    return report


def _place_side(
    state: LiveMakingState,
    clob: ClobClient,
    qm: QuotableMarket,
    quote: Quote,
    pair: QuotePair,
    report: LiveTickReport,
    *,
    dry_run: bool,
    decision_log_path: str,
    cluster_tags: dict,
) -> str | None:
    """Place one leg of a quote pair. A "bid" rests a YES buy at `quote.price`;
    an "ask" is economically a NO buy at `1 - quote.price` -- see the
    _SIDE_TO_ORDER_SIDE docstring for why (no naked shorting). Returns the
    real order id (live), the sentinel `"dry-run"` (dry-run always
    "succeeds"), or `None` on any failure -- callers use this to roll back a
    partially-placed pair rather than leave a naked one-sided order resting
    (see run_live_tick)."""
    if quote.side == "ask":
        if qm.no_token_id is None:
            report.error(f"cannot place ask leg for {qm.market.id}: no NO token id on this market")
            return None
        token_id = qm.no_token_id
        order_price = round_to_tick(1.0 - quote.price, qm.tick_size, mode="nearest")
    else:
        token_id = qm.yes_token_id
        order_price = quote.price

    intent = {
        "market_id": qm.market.id,
        "condition_id": qm.market.condition_id,
        "question": qm.market.question[:70],
        "side": quote.side,
        "token_id": token_id,
        "price": order_price,
        "size_shares": quote.size_shares,
        "collateral_usd": quote.collateral_usd,
    }
    if dry_run:
        report.would_place.append(intent)
        return "dry-run"

    try:
        resp = clob.post_limit_order(
            token_id=token_id,
            price=order_price,
            size=quote.size_shares,
            side=BUY,  # both legs: no naked shorting from flat
            tick_size=qm.tick_size,
            neg_risk=qm.market.neg_risk,
        )
    except Exception as exc:
        report.error(f"place failed for {qm.market.id} {quote.side}: {exc}")
        return None

    order_id = _extract_order_id(resp)
    if not order_id:
        report.error(f"place for {qm.market.id} {quote.side} returned no order id: {resp}")
        return None

    state.add_order(
        LiveOrder(
            order_id=order_id,
            market_id=qm.market.id,
            condition_id=qm.market.condition_id,
            token_id=token_id,
            question=qm.market.question,
            side="buy",
            price=order_price,
            size_shares=quote.size_shares,
            quoted_midpoint=pair.midpoint,
            cluster_tags=cluster_tags,
            tick_size=qm.tick_size,
            neg_risk=qm.market.neg_risk,
        )
    )
    # Stamp tags on the position too (created empty if this is the first
    # time we've ever quoted this token) so cluster exposure is still
    # attributable once the order becomes a fill and the LiveOrder is gone.
    pos = state.position_for(
        qm.market.id,
        condition_id=qm.market.condition_id,
        token_id=token_id,
        question=qm.market.question,
        tick_size=qm.tick_size,
        neg_risk=qm.market.neg_risk,
    )
    if not pos.cluster_tags:
        pos.cluster_tags = cluster_tags

    report.placed.append({**intent, "order_id": order_id})
    log_event({"kind": "place", "order_id": order_id, **intent}, path=decision_log_path)
    return order_id


def _flatten_position(
    state: LiveMakingState,
    clob: ClobClient,
    pos: LiveInventory,
    report: LiveTickReport,
    *,
    dry_run: bool,
    decision_log_path: str,
) -> None:
    """Close out one token's inventory at the touch. Only ever a SELL of
    shares we actually hold (`net_shares` can't go negative under Book M's
    no-naked-shorting placement -- see _place_side), but the buy-if-short
    branch is kept for defensiveness against accounting edge cases."""
    if abs(pos.net_shares) < 1e-9:
        return

    try:
        book = clob.get_order_book(pos.token_id)
    except Exception as exc:
        report.error(f"flatten: could not fetch book for {pos.market_id} ({pos.token_id}): {exc}")
        return

    side = "ask" if pos.net_shares > 0 else "bid"  # sell if long, buy if short
    level = book.best_bid if side == "ask" else book.best_ask
    if level is None:
        report.error(f"flatten: no {'bid' if side == 'ask' else 'ask'} to flatten against for {pos.market_id} ({pos.token_id})")
        return

    intent = {
        "market_id": pos.market_id,
        "token_id": pos.token_id,
        "side": side,
        "price": level.price,
        "size_shares": abs(pos.net_shares),
        "reason": "flatten_before_resolution",
    }
    if dry_run:
        report.would_place.append(intent)
        return

    try:
        resp = clob.post_limit_order(
            token_id=pos.token_id,
            price=level.price,
            size=abs(pos.net_shares),
            side=_SIDE_TO_ORDER_SIDE[side],
            tick_size=pos.tick_size,
            neg_risk=pos.neg_risk,
        )
    except Exception as exc:
        report.error(f"flatten order failed for {pos.market_id} ({pos.token_id}): {exc}")
        return

    order_id = _extract_order_id(resp) or "unknown"
    report.flattened.append({**intent, "order_id": order_id})
    log_event({"kind": "flatten", "order_id": order_id, **intent}, path=decision_log_path)


def _flatten_market(
    state: LiveMakingState,
    clob: ClobClient,
    market_id: str,
    report: LiveTickReport,
    *,
    dry_run: bool,
    decision_log_path: str,
) -> None:
    """Cancel any resting orders and close out any inventory at the touch,
    for a market that has left the quotable universe (typically: it's
    within the flatten-before-resolution window). A market can hold up to
    two inventory rows here -- one per token (YES from a filled bid, NO from
    a filled ask) -- both get closed out independently."""
    orders = state.orders_for(market_id)
    _cancel(state, clob, orders, report, dry_run=dry_run, decision_log_path=decision_log_path)

    for pos in [p for p in state.positions if p.market_id == market_id]:
        _flatten_position(state, clob, pos, report, dry_run=dry_run, decision_log_path=decision_log_path)


def _rank_affordable(quotable: list[QuotableMarket], per_market_budget_usd: float) -> list[QuotableMarket]:
    """task item 1c: at a small live bankroll, most of Phase 0's top-40 by
    raw reward rate are markets we can't afford to quote at all -- a
    symmetric two-sided min-size quote costs exactly `reward.min_size`
    dollars of collateral regardless of price (bid + ask collateral sums to
    `n*price + n*(1-price) = n`), so affordability is a pure min_size vs.
    budget check, no book fetch required.

    Ranks the affordable remainder by daily reward rate per dollar of
    required collateral -- a pool-density proxy, not a competition-adjusted
    share estimate (that needs `making/rewards.py`'s book_qscore, which
    needs a book fetch per candidate; out of scope for a per-tick selection
    pass over the whole scanned universe). Consistent with how Phase 0's own
    `reward_density` sort already ignores competition."""
    affordable = [qm for qm in quotable if qm.reward.min_size <= per_market_budget_usd]
    affordable.sort(key=lambda qm: qm.reward.daily_rate_usd / qm.reward.min_size, reverse=True)
    return affordable


def refresh_universe(
    gamma: GammaClient,
    clob: ClobClient,
    *,
    veto: LLMClassifierVeto | None = None,
    max_gamma_markets: int = GAMMA_MAX_SCAN_MARKETS,
) -> UniverseReport:
    """task item 3: the expensive half of a tick -- Gamma scan + CLOB
    reward-config join + classifier -- split out from run_live_tick so a
    long-running loop (`cmd_make_live_run`) can cache the result and refresh
    it on its own 15-30 min cadence, while run_live_tick's reconcile/unwind/
    quote cycle runs every 30-60s against the cached universe. Measured from
    Phase 0's ticks: a full scan costs 2.5-5 minutes -- running it on every
    fast tick is why quotes were sitting stale for up to the old 30-minute
    loop interval, which is exactly the adverse-selection channel §3.4
    names."""
    veto = veto or NoOpVeto()
    return select_universe(gamma, clob, veto=veto, max_gamma_markets=max_gamma_markets)


def _mark_to_market(state: LiveMakingState, clob: ClobClient, report: LiveTickReport) -> None:
    """task item 7: `LiveInventory` used to be valued at cost basis only, so
    `equity_usd` never moved with price -- the status line read flat while a
    position gained or lost value. Batch-fetches order books for every open
    position's token and stamps `mark_price` from each book's midpoint,
    best-effort per token: a failed fetch just leaves the previous mark (or
    cost basis, if there's never been one -- see `LiveInventory.market_value_usd`)."""
    tokens = [p.token_id for p in state.open_positions]
    if not tokens:
        return
    try:
        books = clob.get_order_books(tokens)
    except Exception as exc:
        report.error(f"mark-to-market: order book batch fetch failed: {exc}")
        return
    for pos in state.open_positions:
        book = books.get(pos.token_id)
        if book is None:
            continue
        bb, ba = book.best_bid, book.best_ask
        if bb and ba:
            pos.mark_price = (bb.price + ba.price) / 2.0


def run_live_tick(
    state: LiveMakingState,
    *,
    universe: UniverseReport,
    gamma: GammaClient | None = None,
    clob: ClobClient | None = None,
    max_markets_quoted: int = MAKING_MAX_MARKETS_QUOTED,
    dry_run: bool = True,
    decision_log_path: str = MAKING_LIVE_DECISION_LOG_FILE,
) -> LiveTickReport:
    """One reconcile/unwind/quote cycle against an already-scanned `universe`
    (see `refresh_universe`) -- this function itself no longer scans Gamma/
    CLOB, so it's cheap enough to run every 30-60s. `gamma` is still used to
    backfill event tags for cluster caps on the few markets that reach
    quoting (`ensure_event_tags`), which is much cheaper than the full scan."""
    gamma = gamma or GammaClient()
    clob = clob or ClobClient()

    report = LiveTickReport(timestamp=_now_iso(), dry_run=dry_run, universe=universe)
    quotable_by_market = {qm.market.id: qm for qm in universe.quotable}
    token_to_market: dict[str, QuotableMarket] = {}
    for qm in universe.quotable:
        token_to_market[qm.yes_token_id] = qm
        if qm.no_token_id:
            token_to_market[qm.no_token_id] = qm

    if not dry_run:
        reconcile_fills(state, clob, report, decision_log_path=decision_log_path, token_to_market=token_to_market)
        compute_markouts(state, clob, report)
        _mark_to_market(state, clob, report)
        check_adverse_selection_kill_switch(state, report)
        check_drawdown_kill_switch(state, report)

    # Unwind anything we're holding (orders or inventory) in a market that
    # has fallen out of the quotable universe -- most commonly because it
    # crossed into the flatten window since the last tick.
    held_market_ids = {o.market_id for o in state.open_orders} | {p.market_id for p in state.open_positions}
    for market_id in held_market_ids:
        if market_id not in quotable_by_market:
            _flatten_market(state, clob, market_id, report, dry_run=dry_run, decision_log_path=decision_log_path)

    if state.halted:
        report.skip("state_halted_no_new_quotes")
        # task item 5: halted must not mean "leave everything resting" --
        # cancel every tracked order before returning, whether this tick is
        # the one that tripped the halt (the kill-switch checks above ran
        # earlier in this same call) or a later tick that's still halted.
        cancel_report = cancel_all_tracked(state, clob, dry_run=dry_run, decision_log_path=decision_log_path)
        report.cancelled.extend(cancel_report.cancelled)
        report.would_cancel.extend(cancel_report.would_cancel)
        report.errors.extend(cancel_report.errors)
        return report

    per_market_budget_usd = MAKING_LIVE_MAX_INVENTORY_PER_MARKET_FRACTION * state.bankroll_cap_usd
    selected: list[QuotableMarket] = _rank_affordable(universe.quotable, per_market_budget_usd)[:max_markets_quoted]
    books = _fetch_books(clob, selected, report)

    # Every resting Book M order is a literal BUY (see _SIDE_TO_ORDER_SIDE's
    # docstring) -- for a bid that's `price * size` of YES, for an ask that's
    # `price * size` of NO, where `order.price` is already the NO price
    # (1 - ask_price, converted at placement time in _place_side). Either way
    # the collateral is just price * size; no side-branch needed.
    deployed = state.deployed_collateral_usd
    for order in state.open_orders:
        deployed += order.price * order.size_shares
    budget = state.bankroll_cap_usd

    # §4.3 cluster caps, applied to real exposure. Rebuilt fresh each tick
    # (matching Phase 0's making/engine.py) and seeded from what we're
    # already carrying -- both open positions and resting orders count,
    # same as the `deployed` budget above, since both are real capital
    # commitments to the same cluster.
    cluster_tracker = ClusterExposureTracker(
        bankroll=state.bankroll_cap_usd,
        entity_cap_fraction=MAKING_LIVE_MAX_ENTITY_CLUSTER_FRACTION,
        theme_cap_fraction=MAKING_LIVE_MAX_THEME_CLUSTER_FRACTION,
        date_bucket_cap_fraction=MAKING_LIVE_MAX_DATE_BUCKET_FRACTION,
        source_cap_fraction=MAKING_LIVE_MAX_RESOLUTION_SOURCE_FRACTION,
    )
    for pos in state.open_positions:
        if pos.cluster_tags:
            cluster_tracker.register(pos.cluster, pos.collateral_usd)
    for order in state.open_orders:
        if order.cluster_tags:
            cluster_tracker.register(order.cluster, order.price * order.size_shares)
    tag_cache: dict[str, list[str]] = {}

    for qm in selected:
        book = books.get(qm.yes_token_id)
        if book is None:
            report.skip("no_order_book")
            continue
        bb, ba = book.best_bid, book.best_ask
        if bb is None or ba is None or ba.price <= bb.price:
            report.skip("book_not_two_sided")
            continue

        existing = state.orders_for(qm.market.id)
        midpoint = (bb.price + ba.price) / 2.0
        stale = any(needs_requote(o.quoted_midpoint, midpoint, MAKING_REQUOTE_MID_MOVE_CENTS) for o in existing)
        if existing and not stale:
            report.skip("already_resting_fresh")
            continue

        # Net YES-equivalent exposure across both legs' tokens -- a NO share
        # is short-YES-equivalent, so it nets against a YES share rather than
        # being tracked as unrelated inventory (see live_state.py's
        # per-(market, token) position keying).
        yes_pos = state.position_for_token(qm.market.id, qm.yes_token_id)
        no_pos = state.position_for_token(qm.market.id, qm.no_token_id) if qm.no_token_id else None
        net_shares = (yes_pos.net_shares if yes_pos else 0.0) - (no_pos.net_shares if no_pos else 0.0)
        cap_shares = _inventory_cap_shares(
            state.bankroll_cap_usd, midpoint, fraction=MAKING_LIVE_MAX_INVENTORY_PER_MARKET_FRACTION
        )
        target_shares = qm.reward.min_size
        if target_shares > cap_shares:
            report.skip("reward_min_size_exceeds_inventory_cap")
            continue

        pair = build_quote_pair(
            market_id=qm.market.id,
            question=qm.market.question,
            token_id=qm.yes_token_id,
            best_bid=bb.price,
            best_ask=ba.price,
            tick_size=qm.tick_size,
            reward=qm.reward,
            target_size_shares=target_shares,
            net_inventory_shares=net_shares,
            inventory_cap_shares=cap_shares,
        )
        if pair.is_empty:
            report.skip("no_eligible_quote")
            continue

        # task item 1d (confirmed: accept + measure): min_size is quoted
        # exactly, so full inventory skew fully suppresses the adding side
        # rather than sizing it down -- a one-sided quote that scores zero on
        # that side. Not avoided structurally (would cost ~half the reachable
        # market count at a small bankroll); tracked instead so `make live
        # report` can surface how much time each market spends one-sided.
        skew_suppressed = (net_shares != 0.0) and (
            "bid_below_reward_min_size" in pair.suppressed or "ask_below_reward_min_size" in pair.suppressed
        )
        if skew_suppressed:
            state.one_sided_ticks[qm.market.id] = state.one_sided_ticks.get(qm.market.id, 0) + 1

        if deployed + pair.collateral_usd > budget:
            report.skip("live_budget_exhausted")
            continue

        ensure_event_tags(qm.market, gamma, tag_cache)
        tags = tag_market(qm.market)
        if cluster_tracker.would_breach(tags, pair.collateral_usd):
            report.skip("cluster_cap")
            continue
        tags_dict = {
            "entity": tags.entity,
            "themes": list(tags.themes),
            "geography": tags.geography,
            "resolution_source": tags.resolution_source,
            "date_bucket": tags.date_bucket,
        }

        if existing and stale:
            _cancel(state, clob, existing, report, dry_run=dry_run, decision_log_path=decision_log_path)

        results: list[str] = []
        any_failed = False
        for quote in (pair.bid, pair.ask):
            if quote is None:
                continue
            if quote.collateral_usd < MAKING_LIVE_MIN_NOTIONAL_USD:
                # task item 5 "min notional": e.g. 20 shares @ $0.03 is
                # $0.60, under Polymarket's ~$1 minimum order notional --
                # skip this leg like a suppressed one rather than attempt a
                # placement that will just be rejected.
                report.skip("below_min_notional")
                continue
            result = _place_side(
                state, clob, qm, quote, pair, report, dry_run=dry_run, decision_log_path=decision_log_path, cluster_tags=tags_dict
            )
            if result is not None:
                results.append(result)
            else:
                any_failed = True

        if not results:
            continue

        if any_failed and not dry_run:
            # task item 5: one leg placed, the other failed -- rolling back
            # the successful leg rather than leaving a naked one-sided
            # resting order (the exact failure mode item 2 fixed for the
            # ask-as-SELL case; this is the same principle applied to a
            # partial-placement error instead of a routing bug).
            real_order_ids = {r for r in results if r != "dry-run"}
            rollback = [o for o in state.open_orders if o.order_id in real_order_ids]
            _cancel(state, clob, rollback, report, dry_run=False, decision_log_path=decision_log_path)
            report.error(f"{qm.market.id}: paired placement failed, rolled back {len(rollback)} leg(s)")
        else:
            deployed += pair.collateral_usd
            cluster_tracker.register(tags, pair.collateral_usd)

    state.n_ticks += 1
    state.last_reconciled_at = _now_iso()
    return report
