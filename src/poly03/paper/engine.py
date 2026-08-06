"""Phase 1 (§8) paper trading: one `run_tick()` call = one scan cadence.

Reuses every Phase 0 module unchanged (filters, classifier, scoring,
sizing, cluster tagging) and adds the parts the README's "Known gaps"
section flagged as missing for Phase 0: §3.3 falsification watch (approximated
-- see below), §5 execution (simulated), §6 lifecycle, §4.4 kill switches.

Two deliberate simplifications, stated once here:

1. **Fill model.** Book A is maker-only (§5.1): we post a limit order at
   the current best bid and, per §2.3, treat that bid as "the price we'd
   actually get filled at as a maker." Paper trading has no real matching
   engine to arrive at a fill probability from, so entries are assumed to
   fill in full at that price whenever the market survives the filters and
   sizing/cap checks. This is optimistic by construction. It is also
   explicitly fine at this phase: §8 says outright "Paper trading cannot
   measure adverse selection or fill quality... Do not skip phase 2" --
   fill-rate measurement is a Phase 2 (micro-live) concern, not a Phase 1
   one. The measurement module flags this so nobody mistakes a paper fill
   rate of 100% for a real one.

2. **Falsification watch (§3.3).** No news/feed monitoring is wired up.
   The proxy used here is cheap but faithful to the doc's own fallback
   logic: re-run the classifier against the market's *current* text/state
   each tick (catches resolution-rules amendments and anything the rules
   engine would now read differently) and treat any adverse price move
   past the §6.2 threshold as "the market telling us something we don't
   know" rather than noise -- which is exactly what §6.2 prescribes when a
   real catalyst feed isn't available.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from poly03.classifier.llm_veto import LLMClassifierVeto, NoOpVeto
from poly03.classifier.pipeline import classify_market
from poly03.classifier.rules import Classification
from poly03.classifier.taxonomy import Tier
from poly03.cluster.tagging import ClusterExposureTracker, tag_market
from poly03.config import (
    BOOK_A_PRICE_BAND,
    EARLY_EXIT_ADVERSE_MOVE_CENTS,
    KILL_DRAWDOWN_FRACTION,
    KILL_LOSS_RATE_MULTIPLE,
    KILL_LOSS_RATE_TRAILING_N,
    KILL_LOSS_WINDOW_DAYS,
    KILL_MAX_LOSSES_IN_WINDOW,
    MAX_CONCURRENT_POSITIONS,
    MIN_MARGIN_PP,
    PAPER_DECISION_LOG_FILE,
    PAPER_MAX_NEW_POSITIONS_PER_TICK,
    PAPER_TARGET_SCAN_MARKETS,
)
from poly03.data.clob import ClobClient
from poly03.data.gamma import GammaClient
from poly03.data.models import Market
from poly03.filters.exclusion import apply_exclusion_filters
from poly03.paper.state import PaperPosition, PaperState, log_decision
from poly03.scoring.edge_score import EdgeScoreInputs, EdgeScoreResult, compute_edge_score
from poly03.scoring.roc import annualized_roc
from poly03.sizing.position_sizing import SizingInputs, check_portfolio_caps, compute_stake

logger = logging.getLogger("poly03.paper")

UNSETTLED_UMA_STATUSES = {"proposed", "disputed", "challenged"}


@dataclass
class CandidateInfo:
    market: Market
    classification: Classification
    maker_price: float
    days: float
    edge: EdgeScoreResult


@dataclass
class TickReport:
    timestamp: str
    scanned: int = 0
    candidates_found: int = 0
    entered: list[str] = field(default_factory=list)
    exited: list[tuple[str, str]] = field(default_factory=list)
    resolved_win: int = 0
    resolved_loss: int = 0
    halted: bool = False
    halt_reasons: list[str] = field(default_factory=list)
    cash: float = 0.0
    equity: float = 0.0


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- §2 / §2.3 scan ---------------------------------------------------------------


def scan_universe(
    gamma: GammaClient,
    *,
    veto: LLMClassifierVeto,
    max_markets: int,
    target_size_usd: float,
    exclude_market_ids: set[str],
    log_path: str,
) -> list[CandidateInfo]:
    """Same pipeline as `poly03 scan`: exclusion filters -> classifier ->
    edge score. Logs every rejection for the §7 counterfactual log."""
    lo, hi = BOOK_A_PRICE_BAND
    candidates: list[CandidateInfo] = []
    scanned = 0

    # Per-market volume order, not iter_markets_with_event_context's per-event
    # order: paginating by event volume front-loads the scan on a handful of
    # mega multi-outcome events (100+ long-shot candidate legs each), which
    # crowds out the 0.85-0.97 price band entirely before the budget runs out.
    # Event tags (only needed for cluster tagging at entry) are backfilled
    # lazily in _enter_new_positions for the few markets that get that far.
    for market in gamma.iter_markets(closed=False, order="volume", ascending=False):
        if scanned >= max_markets:
            break
        scanned += 1

        if market.id in exclude_market_ids:
            continue
        if market.best_bid is None or market.best_ask is None:
            continue
        maker_price = market.best_bid
        if not (lo <= maker_price <= hi):
            continue

        exclusion = apply_exclusion_filters(market)
        if exclusion.excluded:
            log_decision(
                {
                    "kind": "reject",
                    "market_id": market.id,
                    "question": market.question,
                    "reasons": [r.value for r in exclusion.reasons],
                },
                path=log_path,
            )
            continue

        classification = classify_market(market, veto)
        if classification.tier == Tier.TIER_4:
            log_decision(
                {
                    "kind": "reject",
                    "market_id": market.id,
                    "question": market.question,
                    "reasons": ["tier_4_excluded"],
                    "evidence": classification.evidence,
                },
                path=log_path,
            )
            continue

        days = market.days_to_resolution
        if days is None or days <= 0:
            continue

        q_placeholder = min(0.999, maker_price + MIN_MARGIN_PP)
        result = compute_edge_score(
            EdgeScoreInputs(
                market_id=market.id,
                maker_price=maker_price,
                days_to_resolution=days,
                confidence_multiplier=classification.confidence_multiplier,
                target_size_usd=target_size_usd,
                visible_depth_usd=market.liquidity or 0.0,
                estimated_true_probability=q_placeholder,
            )
        )
        if not result.tradeable:
            log_decision(
                {
                    "kind": "reject",
                    "market_id": market.id,
                    "question": market.question,
                    "reasons": ["edge_score_not_tradeable"],
                    "margin_pp": result.margin_pp,
                },
                path=log_path,
            )
            continue

        candidates.append(
            CandidateInfo(market=market, classification=classification, maker_price=maker_price, days=days, edge=result)
        )

    candidates.sort(key=lambda c: c.edge.edge_score, reverse=True)
    return candidates


# --- §6 position lifecycle --------------------------------------------------------


def _evaluate_open_position(
    pos: PaperPosition, gamma: GammaClient, veto: LLMClassifierVeto, hurdle_roc: float | None
) -> tuple[str, str, float] | None:
    """Returns (new_status, reason, close_price) if the position should
    close this tick, else None. Checks are ordered most- to least-urgent:
    resolution, dispute, falsification/tier-downgrade, adverse price move,
    capital recycling."""
    try:
        market = gamma.get_market(pos.market_id)
    except Exception as exc:
        logger.warning("failed to refresh market=%s: %s", pos.market_id, exc)
        return None

    winning_idx = market.winning_outcome_index
    if winning_idx is not None:
        won = winning_idx == pos.side_index
        return ("resolved_win" if won else "resolved_loss", "resolution", 1.0 if won else 0.0)

    if market.closed:
        # closed but not cleanly resolved to 0/1 -- can't keep holding
        fallback_price = market.best_bid if market.best_bid is not None else 0.5
        return ("exited_early", "closed_without_clean_resolution", fallback_price)

    if any(s in UNSETTLED_UMA_STATUSES for s in market.uma_resolution_statuses):
        fallback_price = market.best_bid if market.best_bid is not None else pos.entry_price
        return ("exited_early", "dispute_filed", fallback_price)

    classification = classify_market(market, veto)
    if classification.tier.value > pos.tier or classification.tier == Tier.TIER_4:
        fallback_price = market.best_bid if market.best_bid is not None else pos.entry_price
        return ("exited_early", "tier_downgrade", fallback_price)

    if market.best_bid is not None:
        adverse_cents = (pos.entry_price - market.best_bid) * 100.0
        if adverse_cents > EARLY_EXIT_ADVERSE_MOVE_CENTS:
            return ("exited_early", "adverse_price_move", market.best_bid)

    if hurdle_roc is not None and market.best_ask is not None:
        days_remaining = market.days_to_resolution
        if days_remaining is not None and days_remaining > 0:
            try:
                remaining_roc = annualized_roc(market.best_ask, days_remaining)
            except ValueError:
                remaining_roc = None
            if remaining_roc is not None and remaining_roc < hurdle_roc:
                close_price = market.best_bid if market.best_bid is not None else market.best_ask
                return ("exited_early", "capital_recycling", close_price)

    return None


def _manage_open_positions(
    state: PaperState, gamma: GammaClient, veto: LLMClassifierVeto, hurdle_roc: float | None, log_path: str
) -> tuple[list[tuple[str, str]], int, int]:
    exited: list[tuple[str, str]] = []
    resolved_win = 0
    resolved_loss = 0

    for pos in list(state.open_positions):
        decision = _evaluate_open_position(pos, gamma, veto, hurdle_roc)
        if decision is None:
            continue
        status, reason, close_price = decision
        state.close_position(pos, status=status, reason=reason, close_price=close_price)
        exited.append((pos.id, reason))
        if status == "resolved_win":
            resolved_win += 1
        elif status == "resolved_loss":
            resolved_loss += 1
        log_decision(
            {
                "kind": "exit",
                "position_id": pos.id,
                "market_id": pos.market_id,
                "question": pos.question,
                "status": status,
                "reason": reason,
                "close_price": close_price,
                "realized_pnl": pos.realized_pnl,
            },
            path=log_path,
        )

    return exited, resolved_win, resolved_loss


# --- §4.4 kill switches ------------------------------------------------------------


def check_kill_switches(state: PaperState) -> tuple[bool, list[str], bool]:
    reasons: list[str] = []
    manual_review = False

    if state.high_water_mark > 0:
        drawdown = (state.high_water_mark - state.equity) / state.high_water_mark
        if drawdown > KILL_DRAWDOWN_FRACTION:
            reasons.append(f"bankroll drawdown {drawdown:.1%} > {KILL_DRAWDOWN_FRACTION:.0%} from high-water mark")

    resolutions = [p for p in state.closed_positions if p.status in ("resolved_win", "resolved_loss")]
    trailing = resolutions[-KILL_LOSS_RATE_TRAILING_N:]
    if len(trailing) >= 10:
        implied_loss_rate = sum(1 - p.entry_price for p in trailing) / len(trailing)
        realized_loss_rate = sum(1 for p in trailing if p.status == "resolved_loss") / len(trailing)
        if implied_loss_rate > 0 and realized_loss_rate > implied_loss_rate * KILL_LOSS_RATE_MULTIPLE:
            reasons.append(
                f"realized loss rate {realized_loss_rate:.1%} > {KILL_LOSS_RATE_MULTIPLE}x "
                f"implied {implied_loss_rate:.1%} over trailing {len(trailing)} resolutions"
            )

    cutoff = datetime.now(timezone.utc) - timedelta(days=KILL_LOSS_WINDOW_DAYS)
    recent_losses = [
        p for p in state.closed_positions if p.status == "resolved_loss" and p.closed_at and datetime.fromisoformat(p.closed_at) >= cutoff
    ]
    if len(recent_losses) >= KILL_MAX_LOSSES_IN_WINDOW:
        reasons.append(f"{len(recent_losses)} losses in trailing {KILL_LOSS_WINDOW_DAYS}d window")

    tier1_misses = [p for p in state.closed_positions if p.tier == 1 and p.status == "resolved_loss"]
    if tier1_misses:
        reasons.append(
            f"{len(tier1_misses)} Tier 1 position(s) resolved against us -- classifier is broken, not unlucky"
        )
        manual_review = True

    return (len(reasons) > 0, reasons, manual_review)


# --- entries -----------------------------------------------------------------------


def _ensure_event_tags(market: Market, gamma: GammaClient, tag_cache: dict[str, list[str]]) -> None:
    """scan_universe's iter_markets() pass has no event tags attached.
    Backfill them here, one /events fetch per distinct event_id, only for
    markets that actually reach entry -- not for all 300 scanned."""
    if market.tags or not market.event_id:
        return
    if market.event_id not in tag_cache:
        try:
            tag_cache[market.event_id] = gamma.get_event(market.event_id).tags
        except Exception as exc:
            logger.warning("failed to fetch event tags for event=%s: %s", market.event_id, exc)
            tag_cache[market.event_id] = []
    market.tags = tag_cache[market.event_id]


def _enter_new_positions(
    state: PaperState,
    candidates: list[CandidateInfo],
    cluster_tracker: ClusterExposureTracker,
    gamma: GammaClient,
    log_path: str,
) -> list[str]:
    entered: list[str] = []
    open_market_ids = {p.market_id for p in state.open_positions}
    tag_cache: dict[str, list[str]] = {}

    for cand in candidates:
        if len(entered) >= PAPER_MAX_NEW_POSITIONS_PER_TICK:
            break
        if len(state.open_positions) >= MAX_CONCURRENT_POSITIONS:
            break
        market = cand.market
        if market.id in open_market_ids:
            continue

        outcome = market.outcomes[0] if market.outcomes else "Yes"
        token_id = market.clob_token_ids[0] if market.clob_token_ids else None
        if token_id is None:
            continue

        bankroll = state.equity
        sizing = compute_stake(
            SizingInputs(
                bankroll=bankroll,
                tier=cand.classification.tier,
                maker_price=cand.maker_price,
                estimated_true_probability=min(0.999, cand.maker_price + MIN_MARGIN_PP),
                visible_book_depth_usd=market.liquidity or 0.0,
            )
        )
        stake = sizing.stake_usd
        if stake < market.order_min_size or stake <= 0:
            log_decision(
                {"kind": "reject", "market_id": market.id, "question": market.question, "reasons": ["stake_below_min_order_size"]},
                path=log_path,
            )
            continue
        if stake > state.cash:
            log_decision(
                {"kind": "reject", "market_id": market.id, "question": market.question, "reasons": ["insufficient_cash"]},
                path=log_path,
            )
            continue

        caps = check_portfolio_caps(
            book="A",
            bankroll=bankroll,
            cash_available=state.cash,
            book_a_deployed_usd=state.book_a_deployed_usd,
            book_b_deployed_usd=state.book_b_deployed_usd,
            proposed_stake_usd=stake,
        )
        if not caps.ok:
            log_decision(
                {"kind": "reject", "market_id": market.id, "question": market.question, "reasons": caps.reasons},
                path=log_path,
            )
            continue

        _ensure_event_tags(market, gamma, tag_cache)
        tags = tag_market(market)
        cluster_tracker.update_bankroll(bankroll)
        breaches = cluster_tracker.would_breach(tags, stake)
        if breaches:
            log_decision(
                {"kind": "reject", "market_id": market.id, "question": market.question, "reasons": breaches},
                path=log_path,
            )
            continue

        pos = state.new_position(
            market_id=market.id,
            question=market.question,
            token_id=token_id,
            outcome=outcome,
            side_index=0,
            tier=int(cand.classification.tier),
            entry_price=cand.maker_price,
            stake_usd=stake,
            end_date=market.end_date.isoformat() if market.end_date else None,
            days_to_resolution_at_entry=cand.days,
            modeled_annualized_roc=cand.edge.annualized_roc,
            cluster_tags=tags,
        )
        cluster_tracker.register(tags, stake)
        open_market_ids.add(market.id)
        entered.append(pos.id)
        log_decision(
            {
                "kind": "entry",
                "position_id": pos.id,
                "market_id": market.id,
                "question": market.question,
                "tier": pos.tier,
                "entry_price": pos.entry_price,
                "stake_usd": pos.stake_usd,
                "edge_score": cand.edge.edge_score,
                "binding_constraint": sizing.binding_constraint,
            },
            path=log_path,
        )

    return entered


def _rebuild_cluster_tracker(state: PaperState) -> ClusterExposureTracker:
    tracker = ClusterExposureTracker(bankroll=state.equity)
    for pos in state.open_positions:
        tracker.register(pos.cluster, pos.stake_usd)
    return tracker


# --- entry point ---------------------------------------------------------------------


def run_tick(
    state: PaperState,
    *,
    gamma: GammaClient | None = None,
    clob: ClobClient | None = None,
    veto: LLMClassifierVeto | None = None,
    max_markets: int = PAPER_TARGET_SCAN_MARKETS,
    target_size_hint: float = 200.0,
    decision_log_path: str = PAPER_DECISION_LOG_FILE,
) -> TickReport:
    gamma = gamma or GammaClient()
    clob = clob or ClobClient()  # not currently read from; kept for callers/tests that want to stub it
    veto = veto or NoOpVeto()

    report = TickReport(timestamp=_now_iso())
    state.n_ticks += 1

    open_market_ids = {p.market_id for p in state.open_positions}
    candidates = scan_universe(
        gamma,
        veto=veto,
        max_markets=max_markets,
        target_size_usd=target_size_hint,
        exclude_market_ids=open_market_ids,
        log_path=decision_log_path,
    )
    report.scanned = max_markets
    report.candidates_found = len(candidates)
    hurdle_roc = candidates[0].edge.annualized_roc if candidates else None

    exited, resolved_win, resolved_loss = _manage_open_positions(state, gamma, veto, hurdle_roc, decision_log_path)
    report.exited = exited
    report.resolved_win = resolved_win
    report.resolved_loss = resolved_loss

    halted, halt_reasons, manual_review = check_kill_switches(state)
    state.halted = halted
    state.halt_reasons = halt_reasons
    if manual_review:
        state.manual_review_required = True
    report.halted = state.halted or state.manual_review_required
    report.halt_reasons = halt_reasons

    if not report.halted:
        tracker = _rebuild_cluster_tracker(state)
        report.entered = _enter_new_positions(state, candidates, tracker, gamma, decision_log_path)

    report.cash = state.cash
    report.equity = state.equity
    return report
