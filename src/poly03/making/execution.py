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
    MAKING_LIVE_KILL_MARKOUT_CENTS_PER_SHARE,
    MAKING_LIVE_KILL_MARKOUT_CONSECUTIVE,
    MAKING_MAX_MARKETS_QUOTED,
    MAKING_REQUOTE_MID_MOVE_CENTS,
)
from poly03.data.clob import ClobClient
from poly03.data.gamma import GammaClient
from poly03.making.engine import _fetch_books, _inventory_cap_shares
from poly03.making.live_state import LiveMakingState, LiveOrder, log_event
from poly03.making.quoting import Quote, QuotePair, build_quote_pair, needs_requote
from poly03.making.universe import QuotableMarket, UniverseReport, select_universe

try:
    from py_clob_client.order_builder.constants import BUY, SELL
except ImportError:  # pragma: no cover - py-clob-client always installs this
    BUY, SELL = "BUY", "SELL"

logger = logging.getLogger("poly03.making.execution")

_SIDE_TO_ORDER_SIDE = {"bid": BUY, "ask": SELL}
_ORDER_SIDE_TO_FILL_SIDE = {"bid": "buy", "ask": "sell"}
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


def reconcile_fills(state: LiveMakingState, clob: ClobClient, report: LiveTickReport, *, decision_log_path: str) -> None:
    """Trust the API, not local assumptions (v1 §5.3): pull every open order
    we actually have resting, in one bulk call, rather than trusting that
    `state.open_orders` is still accurate. Diff each tracked order against
    that authoritative list to turn new matched size into a recorded fill,
    drop orders that are no longer open, and flag any order the exchange
    shows that we don't have tracked -- that's a drift signal (a fill our
    engine didn't see, a manual action, a bug), never auto-cancelled here."""
    try:
        remote_orders = clob.get_open_orders()
    except Exception as exc:
        report.error(f"reconcile: get_open_orders failed, falling back to per-order lookups: {exc}")
        remote_orders = None

    fee_cache: dict[str, int | None] = {}

    if remote_orders is not None:
        remote_by_id = {r.get("id"): r for r in remote_orders if r.get("id")}
        tracked_ids = {o.order_id for o in state.open_orders}
        untracked = set(remote_by_id) - tracked_ids
        for order_id in untracked:
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
            _apply_order_update(state, order, remote, report, fee_cache, clob, decision_log_path)
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
            _apply_order_update(state, order, remote, report, fee_cache, clob, decision_log_path)


def _apply_order_update(
    state: LiveMakingState,
    order: LiveOrder,
    remote: dict,
    report: LiveTickReport,
    fee_cache: dict[str, int | None],
    clob: ClobClient,
    decision_log_path: str,
) -> None:
    matched = _remote_size_matched(remote)
    delta = matched - order.size_matched
    if delta > 1e-9:
        fee = _fee_usd(clob, order.token_id, order.price, delta, fee_cache)
        fill = state.record_fill(
            market_id=order.market_id,
            condition_id=order.condition_id,
            token_id=order.token_id,
            question=order.question,
            side=order.side,
            price=order.price,
            size_shares=delta,
            order_id=order.order_id,
            mid_price_at_fill=order.quoted_midpoint,
            fee_usd=fee,
        )
        report.new_fills.append(fill.__dict__)
        log_event({"kind": "fill", **fill.__dict__}, path=decision_log_path)
        order.size_matched = matched

    if _remote_status(remote) in _CLOSED_STATUSES or matched >= order.size_shares - 1e-9:
        state.remove_order(order.order_id)


def compute_markouts(state: LiveMakingState, clob: ClobClient, report: LiveTickReport) -> None:
    """§3.4 adverse selection, tracked directly: for each fill old enough to
    score, fetch the market's current midpoint and mark the fill out
    against it. Signed so positive = price moved in our favor after the
    fill (we bought and it went up, or sold and it went down) and negative
    = adverse selection. `None` stays `None` until a fill is actually old
    enough -- never backfilled with a guess."""
    now = datetime.now(timezone.utc)
    mid_cache: dict[str, float | None] = {}

    def current_mid(token_id: str) -> float | None:
        if token_id not in mid_cache:
            try:
                book = clob.get_order_book(token_id)
                bb, ba = book.best_bid, book.best_ask
                mid_cache[token_id] = (bb.price + ba.price) / 2.0 if bb and ba else None
            except Exception:
                mid_cache[token_id] = None
        return mid_cache[token_id]

    for fill in state.fills:
        if fill.markout_5m_usd is not None and fill.markout_30m_usd is not None:
            continue
        age_minutes = (now - datetime.fromisoformat(fill.filled_at)).total_seconds() / 60.0
        if age_minutes < 5.0:
            continue
        mid = current_mid(fill.token_id)
        if mid is None:
            continue
        sign = 1.0 if fill.side == "buy" else -1.0
        markout = sign * (mid - fill.price) * fill.size_shares
        if fill.markout_5m_usd is None and age_minutes >= 5.0:
            fill.markout_5m_usd = markout
        if fill.markout_30m_usd is None and age_minutes >= 30.0:
            fill.markout_30m_usd = markout


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
) -> None:
    intent = {
        "market_id": qm.market.id,
        "condition_id": qm.market.condition_id,
        "question": qm.market.question[:70],
        "side": quote.side,
        "price": quote.price,
        "size_shares": quote.size_shares,
        "collateral_usd": quote.collateral_usd,
    }
    if dry_run:
        report.would_place.append(intent)
        return

    try:
        resp = clob.post_limit_order(
            token_id=qm.yes_token_id,
            price=quote.price,
            size=quote.size_shares,
            side=_SIDE_TO_ORDER_SIDE[quote.side],
            tick_size=qm.tick_size,
            neg_risk=qm.market.neg_risk,
        )
    except Exception as exc:
        report.error(f"place failed for {qm.market.id} {quote.side}: {exc}")
        return

    order_id = _extract_order_id(resp)
    if not order_id:
        report.error(f"place for {qm.market.id} {quote.side} returned no order id: {resp}")
        return

    state.add_order(
        LiveOrder(
            order_id=order_id,
            market_id=qm.market.id,
            condition_id=qm.market.condition_id,
            token_id=qm.yes_token_id,
            question=qm.market.question,
            side=_ORDER_SIDE_TO_FILL_SIDE[quote.side],
            price=quote.price,
            size_shares=quote.size_shares,
            quoted_midpoint=pair.midpoint,
            cluster_tags=cluster_tags,
        )
    )
    # Stamp tags on the position too (created empty if this is the first
    # time we've ever quoted this market) so cluster exposure is still
    # attributable once the order becomes a fill and the LiveOrder is gone.
    pos = state.position_for(
        qm.market.id, condition_id=qm.market.condition_id, token_id=qm.yes_token_id, question=qm.market.question
    )
    if not pos.cluster_tags:
        pos.cluster_tags = cluster_tags

    report.placed.append({**intent, "order_id": order_id})
    log_event({"kind": "place", "order_id": order_id, **intent}, path=decision_log_path)


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
    within the flatten-before-resolution window)."""
    orders = state.orders_for(market_id)
    _cancel(state, clob, orders, report, dry_run=dry_run, decision_log_path=decision_log_path)

    pos = next((p for p in state.positions if p.market_id == market_id), None)
    if pos is None or abs(pos.net_shares) < 1e-9:
        return

    try:
        book = clob.get_order_book(pos.token_id)
    except Exception as exc:
        report.error(f"flatten: could not fetch book for {market_id}: {exc}")
        return

    side = "ask" if pos.net_shares > 0 else "bid"  # sell if long, buy if short
    level = book.best_bid if side == "ask" else book.best_ask
    if level is None:
        report.error(f"flatten: no {'bid' if side == 'ask' else 'ask'} to flatten against for {market_id}")
        return

    intent = {
        "market_id": market_id,
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
            tick_size=0.01,
            neg_risk=False,
        )
    except Exception as exc:
        report.error(f"flatten order failed for {market_id}: {exc}")
        return

    order_id = _extract_order_id(resp) or "unknown"
    report.flattened.append({**intent, "order_id": order_id})
    log_event({"kind": "flatten", "order_id": order_id, **intent}, path=decision_log_path)


def run_live_tick(
    state: LiveMakingState,
    *,
    gamma: GammaClient | None = None,
    clob: ClobClient | None = None,
    veto: LLMClassifierVeto | None = None,
    max_gamma_markets: int = GAMMA_MAX_SCAN_MARKETS,
    max_markets_quoted: int = MAKING_MAX_MARKETS_QUOTED,
    dry_run: bool = True,
    decision_log_path: str = MAKING_LIVE_DECISION_LOG_FILE,
) -> LiveTickReport:
    gamma = gamma or GammaClient()
    clob = clob or ClobClient()
    veto = veto or NoOpVeto()

    universe = select_universe(gamma, clob, veto=veto, max_gamma_markets=max_gamma_markets)
    report = LiveTickReport(timestamp=_now_iso(), dry_run=dry_run, universe=universe)
    quotable_by_market = {qm.market.id: qm for qm in universe.quotable}

    if not dry_run:
        reconcile_fills(state, clob, report, decision_log_path=decision_log_path)
        compute_markouts(state, clob, report)
        check_adverse_selection_kill_switch(state, report)

    # Unwind anything we're holding (orders or inventory) in a market that
    # has fallen out of the quotable universe -- most commonly because it
    # crossed into the flatten window since the last tick.
    held_market_ids = {o.market_id for o in state.open_orders} | {p.market_id for p in state.open_positions}
    for market_id in held_market_ids:
        if market_id not in quotable_by_market:
            _flatten_market(state, clob, market_id, report, dry_run=dry_run, decision_log_path=decision_log_path)

    if state.halted:
        report.skip("state_halted_no_new_quotes")
        return report

    selected: list[QuotableMarket] = universe.quotable[:max_markets_quoted]
    books = _fetch_books(clob, selected, report)

    deployed = state.deployed_collateral_usd
    for order in state.open_orders:
        deployed += order.price * order.size_shares if order.side == "buy" else (1.0 - order.price) * order.size_shares
    budget = state.bankroll_cap_usd

    # §4.3 cluster caps, applied to real exposure. Rebuilt fresh each tick
    # (matching Phase 0's making/engine.py) and seeded from what we're
    # already carrying -- both open positions and resting orders count,
    # same as the `deployed` budget above, since both are real capital
    # commitments to the same cluster.
    cluster_tracker = ClusterExposureTracker(bankroll=state.bankroll_cap_usd)
    for pos in state.open_positions:
        if pos.cluster_tags:
            cluster_tracker.register(pos.cluster, pos.collateral_usd)
    for order in state.open_orders:
        if order.cluster_tags:
            order_collateral = order.price * order.size_shares if order.side == "buy" else (1.0 - order.price) * order.size_shares
            cluster_tracker.register(order.cluster, order_collateral)
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

        pos = next((p for p in state.positions if p.market_id == qm.market.id), None)
        net_shares = pos.net_shares if pos else 0.0
        cap_shares = _inventory_cap_shares(state.bankroll_cap_usd, midpoint)
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

        placed_any = False
        for quote in (pair.bid, pair.ask):
            if quote is None:
                continue
            _place_side(
                state, clob, qm, quote, pair, report, dry_run=dry_run, decision_log_path=decision_log_path, cluster_tags=tags_dict
            )
            placed_any = True
        if placed_any:
            deployed += pair.collateral_usd
            cluster_tracker.register(tags, pair.collateral_usd)

    state.n_ticks += 1
    state.last_reconciled_at = _now_iso()
    return report
