"""CLI entry points for the Phase 0 research stack: `poly03 scan`,
`poly03 backtest`, `poly03 montecarlo`. Nothing here places an order --
Phase 0 is data/backtest only (see strategy_v1.md §8)."""

from __future__ import annotations

import argparse
import logging
import sys

from poly03.classifier.pipeline import classify_market
from poly03.classifier.taxonomy import Tier
from poly03.config import (
    BOOK_A_HORIZON_CAP_DAYS,
    BOOK_A_PRICE_BAND,
    BOOK_A_REQUIRE_EDGE_ESTIMATE,
    GAMMA_MAX_SCAN_MARKETS,
    MAKING_DECISION_LOG_FILE,
    MAKING_LIVE_DECISION_LOG_FILE,
    MAKING_LIVE_STATE_FILE,
    MAKING_MAX_MARKETS_QUOTED,
    MAKING_STATE_FILE,
    MIN_MARGIN_PP,
    PAPER_DECISION_LOG_FILE,
    PAPER_STARTING_BANKROLL,
    PAPER_STATE_FILE,
    PAPER_TARGET_SCAN_MARKETS,
)
from poly03.data.gamma import GammaClient
from poly03.filters.exclusion import apply_exclusion_filters
from poly03.logutil import append_log
from poly03.scoring.edge_score import EdgeScoreInputs, compute_edge_score


def _log(msg: object = "") -> None:
    """print() that also appends to PAPER_TRADE_LOG_FILE, so console output
    from every subcommand survives after the terminal/session is gone."""
    msg = str(msg)
    print(msg, flush=True)  # unbuffered: this loop is meant to run under nohup/tmux
    append_log(msg)


def cmd_scan(args: argparse.Namespace) -> None:
    """Apply exclusion filters + classifier + ROC scoring to the current
    open-market universe and print surviving Book A candidates.

    There is still no model for q here. strategy_v2.md §1.1 is the write-up of
    why that matters: the placeholder `price + MIN_MARGIN_PP` makes the margin
    gate a tautology, so this scan reports "candidates" that were never
    actually screened for edge. It stays available as a universe-inspection
    tool and prints the warning below; `poly03 make scan` is the Book M
    equivalent that doesn't need q at all.
    """
    if BOOK_A_REQUIRE_EDGE_ESTIMATE:
        _log(
            "WARNING: no edge estimator is wired up, so q is a placeholder and the\n"
            "  margin gate below cannot reject anything for lack of edge (§1.1).\n"
            "  These are price-band survivors, not screened candidates.\n"
        )
    gamma = GammaClient()
    lo, hi = BOOK_A_PRICE_BAND
    found = 0
    scanned = 0

    for market in gamma.iter_markets_with_event_context(
        closed=False, order="volume", ascending=False, page_size=args.page_size
    ):
        if scanned >= args.max_markets:
            break
        scanned += 1

        if market.best_bid is None or market.best_ask is None:
            continue
        maker_price = market.best_bid
        if not (lo <= maker_price <= hi):
            continue

        exclusion = apply_exclusion_filters(market)
        if exclusion.excluded:
            continue

        classification = classify_market(market)
        if classification.tier == Tier.TIER_4:
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
                target_size_usd=args.target_size,
                visible_depth_usd=market.liquidity or 0.0,
                estimated_true_probability=q_placeholder,
            )
        )
        if not result.tradeable:
            continue

        found += 1
        _log(
            f"[T{classification.tier.value}] {market.question[:70]:70} "
            f"price={maker_price:.3f} days={days:6.1f} "
            f"roc={result.annualized_roc:+7.1%} edge={result.edge_score:+.3f}"
        )
        _log(f"       evidence: {classification.evidence[0]}")

    _log(f"\nscanned {scanned} markets, {found} tradeable Book A candidates found")


def cmd_backtest(args: argparse.Namespace) -> None:
    from poly03.backtest.engine import run_phase0_backtest

    _log(f"running Phase 0 backtest over up to {args.max_markets} closed markets (this hits the CLOB API a lot)...")
    report = run_phase0_backtest(max_markets=args.max_markets, page_size=args.page_size)

    _log(f"\ntotal candidates found: {len(report.candidates)}")
    _log(f"tradeable (passed filters, tier != 4, in Book A price band): {len(report.tradeable)}")

    _log("\ncalibration by tier (§7 -- the single most important number):")
    for bucket in report.calibration_by_tier():
        _log(
            f"  Tier {bucket.tier.value}: n={bucket.n:4d}  "
            f"mean_entry_price={bucket.mean_entry_price:.3f}  "
            f"realized_win_rate={bucket.realized_win_rate:.1%}  "
            f"brier={bucket.brier_score:.4f}"
        )

    brier = report.overall_brier()
    _log(f"\noverall Brier score: {brier:.4f}" if brier is not None else "\noverall Brier score: n/a (no tradeable candidates)")

    roc = report.realized_vs_modeled_roc()
    if roc:
        modeled, realized = roc
        _log(f"mean modeled annualized ROC:  {modeled:+.1%}")
        _log(f"mean realized annualized ROC: {realized:+.1%}")

    misses = report.tier1_misses()
    _log(f"\nTier 1 misses (kill-switch trigger if any): {len(misses)}")
    for m in misses:
        _log(f"  BROKEN CLASSIFIER: {m.question[:70]} (side={m.side}, entry={m.entry_price:.3f})")

    rejected = report.counterfactual_rejected()
    _log(f"\ncounterfactual: {len(rejected)} candidates rejected by exclusion filters")


def cmd_paper_tick(args: argparse.Namespace) -> None:
    from poly03.paper.engine import run_tick
    from poly03.paper.state import load_state, save_state

    state = load_state(args.state_file)
    if state.manual_review_required and not args.force:
        _log("HALTED: manual_review_required is set (a Tier 1 position resolved against us).")
        _log("Investigate the classifier before continuing. Re-run with --force to tick anyway.")
        return

    report = run_tick(state, max_markets=args.max_markets, decision_log_path=args.log_file)
    save_state(state, args.state_file)

    _log(f"tick @ {report.timestamp}")
    _log(f"  scanned={report.scanned} candidates={report.candidates_found}")
    _log(f"  entered={len(report.entered)} exited={len(report.exited)} (win={report.resolved_win} loss={report.resolved_loss})")
    _log(f"  cash=${report.cash:,.2f} equity=${report.equity:,.2f}")
    if report.halted:
        _log(f"  HALTED: {'; '.join(report.halt_reasons) or 'manual review required'}")


def cmd_paper_status(args: argparse.Namespace) -> None:
    from poly03.paper.state import load_state

    state = load_state(args.state_file)
    _log(f"bankroll (start): ${state.bankroll:,.2f}")
    _log(f"equity (cash + open stake basis): ${state.equity:,.2f}")
    _log(f"cash: ${state.cash:,.2f}")
    _log(f"high-water mark: ${state.high_water_mark:,.2f}")
    _log(f"open positions: {len(state.open_positions)}  closed positions: {len(state.closed_positions)}")
    _log(f"ticks run: {state.n_ticks}")
    if state.halted or state.manual_review_required:
        _log(f"HALTED: {'; '.join(state.halt_reasons)}")
        if state.manual_review_required:
            _log("MANUAL REVIEW REQUIRED (Tier 1 miss)")


def cmd_paper_report(args: argparse.Namespace, log=_log) -> None:
    from poly03.paper import measurement as m
    from poly03.paper.state import load_state

    state = load_state(args.state_file)

    log("calibration by tier (§7 -- the single most important number):")
    for b in m.calibration_by_tier(state):
        log(
            f"  Tier {b.tier}: n={b.n:4d} mean_entry_price={b.mean_entry_price:.3f} "
            f"realized_win_rate={b.realized_win_rate:.1%} brier={b.brier_score:.4f}"
        )

    brier = m.overall_brier(state)
    log(f"\noverall Brier score: {brier:.4f}" if brier is not None else "\noverall Brier score: n/a")

    roc = m.realized_vs_modeled_roc(state)
    if roc:
        modeled, realized = roc
        log(f"mean modeled annualized ROC:  {modeled:+.1%}")
        log(f"mean realized annualized ROC: {realized:+.1%}")

    dd = m.drawdown_stats(state)
    log(f"\nmax drawdown: {dd.max_drawdown_fraction:.1%} (${dd.max_drawdown_usd:,.2f})")
    log(f"current drawdown: {dd.current_drawdown_fraction:.1%}")
    log(f"time to recovery: {dd.time_to_recovery_days:.1f}d" if dd.time_to_recovery_days is not None else "time to recovery: n/a")

    log("\nP&L by tier:")
    for tier, pnl in sorted(m.pnl_by_tier(state).items()):
        log(f"  Tier {tier}: {pnl:+,.2f}")

    log("\nexit reasons:")
    for reason, n in sorted(m.exit_reason_counts(state).items(), key=lambda kv: -kv[1]):
        log(f"  {reason}: {n}")

    log("\nnote: fill rate is not meaningful in paper trading (fills are assumed, not simulated -- see paper/engine.py docstring). Real fill quality is a Phase 2 (micro-live) measurement.")

    log("\n--- §8 Phase 1 gate ---")
    log(m.phase1_gate_status(state).summary())


def cmd_paper_reset(args: argparse.Namespace) -> None:
    from pathlib import Path

    from poly03.config import PAPER_STARTING_BANKROLL
    from poly03.paper.state import PaperState, save_state

    bankroll = args.bankroll if args.bankroll is not None else PAPER_STARTING_BANKROLL
    state = PaperState(bankroll=bankroll, cash=bankroll, high_water_mark=bankroll)
    save_state(state, args.state_file)
    Path(args.log_file).unlink(missing_ok=True)
    _log(f"paper state reset: bankroll=${bankroll:,.2f}  state_file={args.state_file}")


def cmd_paper_run(args: argparse.Namespace) -> None:
    """Run the §8 Phase 1 loop forever: tick on an interval, Ctrl+C to stop
    and print the §7 report. Meant to be left running (screen/tmux/systemd)
    for the doc's 30-60 day paper window -- see strategy_v1.md §8."""
    import time
    from pathlib import Path

    from poly03.notify.telegram import TelegramReporter
    from poly03.paper.engine import run_tick
    from poly03.paper.state import PaperState, load_state, save_state

    logger = logging.getLogger("poly03.paper")
    notifier = TelegramReporter()
    if not notifier.enabled:
        _log("(Telegram not configured -- set TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID in .env to mirror this output there)")

    path = Path(args.state_file)
    if path.exists():
        state = load_state(args.state_file)
        notifier.log(
            f"resuming existing paper state from {args.state_file}: equity=${state.equity:,.2f}, "
            f"{len(state.open_positions)} open / {len(state.closed_positions)} closed positions"
        )
        if args.bankroll is not None and args.bankroll != state.bankroll:
            notifier.log(
                f"NOTE: --bankroll {args.bankroll:,.2f} was given but is ignored because "
                f"{args.state_file} already exists (starting bankroll stays ${state.bankroll:,.2f}). "
                f"Use `paper reset --bankroll {args.bankroll:,.2f}` to change it."
            )
    else:
        from poly03.config import PAPER_STARTING_BANKROLL

        bankroll = args.bankroll if args.bankroll is not None else PAPER_STARTING_BANKROLL
        state = PaperState(bankroll=bankroll, cash=bankroll, high_water_mark=bankroll)
        save_state(state, args.state_file)
        notifier.log(f"started new paper run: bankroll=${bankroll:,.2f}  state_file={args.state_file}")

    notifier.log(f"ticking every {args.interval}s. Press Ctrl+C to stop and print the final report.\n")
    notifier.flush()

    try:
        while True:
            if state.manual_review_required and not args.force:
                notifier.log(
                    "HALTED: manual_review_required is set (a Tier 1 position resolved against us). "
                    "Investigate the classifier, then `paper reset` or re-run with --force."
                )
                notifier.flush()
                break
            try:
                report = run_tick(state, max_markets=args.max_markets, decision_log_path=args.log_file)
                save_state(state, args.state_file)
                status = f"HALTED: {'; '.join(report.halt_reasons)}" if report.halted else "ok"
                notifier.log(
                    f"[{report.timestamp}] scanned={report.scanned} candidates={report.candidates_found} "
                    f"entered={len(report.entered)} exited={len(report.exited)} "
                    f"(win={report.resolved_win} loss={report.resolved_loss}) "
                    f"cash=${report.cash:,.2f} equity=${report.equity:,.2f} [{status}]"
                )
            except Exception as exc:
                logger.warning("tick failed, will retry next interval: %s", exc)
            notifier.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        notifier.log("\nstopping (Ctrl+C received)...")
    finally:
        save_state(state, args.state_file)
        notifier.log("\n=== final §7 report ===\n")
        cmd_paper_report(args, log=notifier.log)
        notifier.flush()


# --- strategy_v2.md §3/§4: Book M -----------------------------------------------


def cmd_make_scan(args: argparse.Namespace) -> None:
    """§3.1: show the current quotable universe and why everything else was
    dropped. Read-only, no state written."""
    from poly03.data.clob import ClobClient
    from poly03.making.universe import select_universe

    report = select_universe(GammaClient(), ClobClient(), max_gamma_markets=args.max_markets)

    _log(f"gamma markets scanned:      {report.scanned}")
    _log(f"reward-eligible (CLOB):     {report.reward_eligible}")
    _log(f"quotable after §3.1 gates:  {len(report.quotable)}\n")

    for qm in report.quotable[: args.limit]:
        _log(
            f"  ${qm.reward.daily_rate_usd:7,.0f}/day  {qm.market.best_bid:.3f}/{qm.market.best_ask:.3f} "
            f"spread={qm.spread:.3f} minsz={qm.reward.min_size:>5,.0f} "
            f"maxsp={qm.reward.max_spread_cents:.1f}c  [T{qm.classification.tier.value}] "
            f"{qm.market.question[:52]}"
        )

    pool = sum(qm.reward.daily_rate_usd for qm in report.quotable)
    _log(f"\ntotal reward pool across quotable markets: ${pool:,.0f}/day (${pool * 365:,.0f}/yr, split among all makers)")

    _log("\nwhy markets were dropped (§3.1):")
    for reason, n in sorted(report.rejections.items(), key=lambda kv: -kv[1]):
        _log(f"  {reason}: {n}")


def cmd_make_tick(args: argparse.Namespace) -> None:
    from poly03.making.engine import run_tick
    from poly03.making.state import load_state, save_state

    state = load_state(args.state_file)
    report = run_tick(
        state,
        max_gamma_markets=args.max_markets,
        max_markets_quoted=args.max_quoted,
        decision_log_path=args.log_file,
    )
    save_state(state, args.state_file)

    s = report.summary
    _log(f"tick @ {report.timestamp}")
    _log(f"  scanned={s.gamma_scanned} reward_eligible={s.reward_eligible} quotable={s.quotable} quoted={s.quoted}")
    _log(f"  collateral=${s.total_collateral_usd:,.2f}  est_rewards=${s.total_est_reward_usd_per_day:,.2f}/day")
    if s.total_collateral_usd > 0:
        _log(f"  implied yield on collateral: {s.daily_yield_on_collateral * 365:.1%}/yr (rewards only, pre-fills)")
    if report.skipped:
        _log(f"  skipped: {', '.join(f'{k}={v}' for k, v in sorted(report.skipped.items(), key=lambda kv: -kv[1]))}")


def cmd_make_report(args: argparse.Namespace, log=_log) -> None:
    from poly03.making import measurement as m
    from poly03.making.state import load_state

    state = load_state(args.state_file)
    log("=== Book M -- §4 Phase 0 measurement ===\n")
    log(m.SCORING_CAVEAT + "\n")

    est = m.reward_estimate(state)
    if est is None:
        log("no ticks recorded yet -- run `poly03 make run` first.")
        return

    log(f"observation window: {est.observation_days:.2f}d over {est.n_ticks} ticks")
    log(f"median collateral deployed:   ${est.median_collateral_usd:,.2f}")
    log(f"median reward pool in quoted markets: ${est.median_pool_usd_per_day:,.2f}/day")
    log(
        f"estimated capture (all):      ${est.median_usd_per_day:,.2f}/day "
        f"(p25 ${est.p25_usd_per_day:,.2f} / p75 ${est.p75_usd_per_day:,.2f})"
    )
    log(f"  of which IDENTIFIED:        ${est.median_identified_usd_per_day:,.2f}/day  <-- the defensible number")
    log(
        f"  of which unidentified:      "
        f"${est.median_usd_per_day - est.median_identified_usd_per_day:,.2f}/day "
        f"across ~{est.median_unidentified_quoted:.0f} markets with no competing depth"
    )
    log(f"estimated share of pool:      {est.median_share_of_pool:.2%}")
    log(f"annualized yield on collateral: {est.annualized_yield_on_collateral:.1%}  <-- identified rewards only")
    log("  (excludes spread capture and the maker rebate, which need fills, and")
    log("   excludes adverse selection, which can exceed all of them -- §3.4)")

    warning = m.implausibility_warning(est)
    if warning:
        log("\n" + warning)

    log("\ntop markets by cumulative estimated accrual:")
    for market_id, accrual in m.top_markets_by_accrual(state):
        log(f"  {market_id}: {accrual:,.2f} reward-USD-days")

    log("\nuniverse funnel (§3.1 rejections, all ticks):")
    for reason, n in m.universe_funnel(state)[:12]:
        log(f"  {reason}: {n:,}")

    log("\n--- §4 Phase 0 gate ---")
    log(m.phase0_gate(state).summary())


def cmd_make_reset(args: argparse.Namespace) -> None:
    from pathlib import Path

    from poly03.making.state import MakingState, save_state

    bankroll = args.bankroll if args.bankroll is not None else PAPER_STARTING_BANKROLL
    save_state(MakingState(bankroll=bankroll), args.state_file)
    Path(args.log_file).unlink(missing_ok=True)
    _log(f"Book M state reset: bankroll=${bankroll:,.2f}  state_file={args.state_file}")


def cmd_make_run(args: argparse.Namespace) -> None:
    """§4 Phase 0: tick on an interval and accumulate the reward-share
    observation series. Ctrl+C stops and prints the report."""
    import time
    from pathlib import Path

    from poly03.making.engine import run_tick
    from poly03.making.state import MakingState, load_state, save_state
    from poly03.notify.telegram import TelegramReporter

    logger = logging.getLogger("poly03.making")
    notifier = TelegramReporter()

    path = Path(args.state_file)
    if path.exists():
        state = load_state(args.state_file)
        notifier.log(f"resuming Book M Phase 0 from {args.state_file}: {state.n_ticks} ticks recorded")
    else:
        bankroll = args.bankroll if args.bankroll is not None else PAPER_STARTING_BANKROLL
        state = MakingState(bankroll=bankroll)
        save_state(state, args.state_file)
        notifier.log(f"started Book M Phase 0: bankroll=${bankroll:,.2f}  state_file={args.state_file}")

    notifier.log(f"ticking every {args.interval}s. No orders are placed. Ctrl+C to stop and report.\n")
    notifier.flush()

    try:
        while True:
            try:
                report = run_tick(
                    state,
                    max_gamma_markets=args.max_markets,
                    max_markets_quoted=args.max_quoted,
                    decision_log_path=args.log_file,
                )
                save_state(state, args.state_file)
                s = report.summary
                notifier.log(
                    f"[{report.timestamp}] quotable={s.quotable} quoted={s.quoted} "
                    f"collateral=${s.total_collateral_usd:,.0f} "
                    f"est_rewards=${s.total_est_reward_usd_per_day:,.2f}/day"
                )
            except Exception as exc:
                logger.warning("tick failed, will retry next interval: %s", exc)
            notifier.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        notifier.log("\nstopping (Ctrl+C received)...")
    finally:
        save_state(state, args.state_file)
        notifier.log("\n=== final §4 Phase 0 report ===\n")
        cmd_make_report(args, log=notifier.log)
        notifier.flush()


# --- strategy_v2.md §4 Phase 1: micro-live execution ---------------------------


def _report_line(report, state) -> str:
    mode = "LIVE" if not report.dry_run else "DRY-RUN"
    if report.dry_run:
        action_bits = f"would_place={len(report.would_place)} would_cancel={len(report.would_cancel)}"
    else:
        action_bits = (
            f"placed={len(report.placed)} cancelled={len(report.cancelled)} "
            f"flattened={len(report.flattened)} new_fills={len(report.new_fills)}"
        )
    line = (
        f"[{mode}] [{report.timestamp}] quotable={len(report.universe.quotable)} {action_bits} "
        f"equity=${state.equity_usd:,.2f}"
    )
    if report.errors:
        line += f" ERRORS={len(report.errors)}"
    return line


def cmd_make_live_tick(args: argparse.Namespace) -> None:
    from poly03.making.execution import run_live_tick
    from poly03.making.live_state import load_state, save_state

    state = load_state(args.state_file)
    if state.halted and not args.force:
        _log("HALTED: Book M live state is halted. Investigate before continuing. Re-run with --force to tick anyway.")
        return

    report = run_live_tick(
        state,
        max_gamma_markets=args.max_markets,
        max_markets_quoted=args.max_quoted,
        dry_run=not args.live,
        decision_log_path=args.log_file,
    )
    save_state(state, args.state_file)

    _log(_report_line(report, state))
    if report.errors:
        for e in report.errors[:5]:
            _log(f"  error: {e}")
    if report.skipped:
        _log(f"  skipped: {', '.join(f'{k}={v}' for k, v in sorted(report.skipped.items(), key=lambda kv: -kv[1]))}")
    if not args.live and report.would_place:
        _log("  (dry-run: pass --live to actually place these)")


def cmd_make_live_status(args: argparse.Namespace) -> None:
    from poly03.making.live_state import load_state

    state = load_state(args.state_file)
    _log(f"bankroll cap: ${state.bankroll_cap_usd:,.2f}")
    _log(f"cash: ${state.cash_usd:,.2f}")
    _log(f"deployed collateral: ${state.deployed_collateral_usd:,.2f}")
    _log(f"equity: ${state.equity_usd:,.2f}")
    _log(f"open positions: {len(state.open_positions)}  open orders: {len(state.open_orders)}  fills: {len(state.fills)}")
    _log(f"realized reward: ${state.realized_reward_usd_total:,.2f} ({len(state.reward_payouts)} logged payouts)")
    _log(f"realized fee: ${state.realized_fee_usd_total:,.2f}")

    scored_5m = [f.markout_5m_usd for f in state.fills if f.markout_5m_usd is not None]
    scored_30m = [f.markout_30m_usd for f in state.fills if f.markout_30m_usd is not None]
    if scored_5m:
        _log(f"adverse selection, 5m markout:  sum=${sum(scored_5m):+,.2f}  n={len(scored_5m)}  avg=${sum(scored_5m) / len(scored_5m):+,.4f}")
    if scored_30m:
        _log(f"adverse selection, 30m markout: sum=${sum(scored_30m):+,.2f}  n={len(scored_30m)}  avg=${sum(scored_30m) / len(scored_30m):+,.4f}")
    if state.fills and not scored_5m:
        _log("adverse selection: no fills old enough yet to score (needs 5m+)")

    _log(f"ticks run: {state.n_ticks}  last reconciled: {state.last_reconciled_at or 'never'}")
    if state.halted:
        _log(f"HALTED: {'; '.join(state.halt_reasons)}")


def cmd_make_live_report(args: argparse.Namespace, log=_log) -> None:
    from pathlib import Path

    from poly03.making import live_measurement as m
    from poly03.making.live_state import load_state as load_live_state
    from poly03.making.state import load_state as load_phase0_state

    state = load_live_state(args.state_file)

    log("=== Book M -- §4 Phase 1 measurement ===\n")

    phase0_state = None
    if Path(args.phase0_state_file).exists():
        phase0_state = load_phase0_state(args.phase0_state_file)
    rc = m.reward_capture_comparison(state, phase0_state)
    log("reward capture, realized vs. Phase 0 estimate:")
    if rc.realized_usd_per_day is not None:
        log(f"  realized:  ${rc.realized_usd_per_day:,.2f}/day (${rc.realized_total_usd:,.2f} total, {rc.n_payouts} logged payouts over {rc.observation_days:.1f}d)")
    else:
        log(f"  realized:  n/a -- observation window too short ({rc.observation_days * 24:.1f}h < 1h) to compute a rate (${rc.realized_total_usd:,.2f} total so far)")
    if rc.estimated_usd_per_day is not None:
        log(f"  estimated: ${rc.estimated_usd_per_day:,.2f}/day (Phase 0, {args.phase0_state_file})")
        if rc.ratio is not None:
            log(f"  ratio (realized/estimated): {rc.ratio:.1%}")
    else:
        log(f"  estimated: n/a (no Phase 0 state found at {args.phase0_state_file})")

    fr = m.fill_rate(args.log_file)
    log("\nfill rate on resting quotes (real, not assumed):")
    log(f"  orders placed: {fr.n_orders_placed}  orders filled (>=1 fill): {fr.n_orders_filled}")
    if fr.order_count_fill_rate is not None:
        log(f"  fill rate by order count: {fr.order_count_fill_rate:.1%}")
    if fr.notional_fill_rate is not None:
        log(f"  fill rate by notional:    {fr.notional_fill_rate:.1%} (${fr.notional_filled_usd:,.2f} / ${fr.notional_placed_usd:,.2f})")

    adv = m.adverse_selection_summary(state)
    log("\nadverse selection (markout-based, 30m where matured else 5m):")
    log(f"  fills: {adv.n_fills}  scored: {adv.n_scored}")
    log(f"  spread capture (favorable markouts): ${adv.spread_capture_usd:,.2f}")
    log(f"  adverse selection (unfavorable markouts): ${adv.adverse_selection_usd:,.2f}")
    log(f"  reward: ${adv.reward_usd:,.2f}   fees: ${adv.fee_usd:,.2f}")
    log(f"  capture (reward + spread capture - fees): ${adv.capture_usd:,.2f}")
    log(f"  net (capture - adverse selection): ${adv.net_usd:+,.2f}")

    log("\n--- §4 Phase 1 gate ---")
    log(m.phase1_gate(state).summary())


def cmd_make_live_record_reward(args: argparse.Namespace) -> None:
    from poly03.making.live_state import load_state, save_state

    state = load_state(args.state_file)
    state.record_reward_payout(args.amount, note=args.note or "")
    save_state(state, args.state_file)
    _log(f"logged reward payout of ${args.amount:,.2f}. realized_reward_usd_total is now ${state.realized_reward_usd_total:,.2f}")


def cmd_make_live_reset(args: argparse.Namespace) -> None:
    from pathlib import Path

    from poly03.config import MAKING_LIVE_BANKROLL_CAP_USD
    from poly03.making.live_state import LiveMakingState, save_state

    cap = args.bankroll_cap if args.bankroll_cap is not None else MAKING_LIVE_BANKROLL_CAP_USD
    save_state(LiveMakingState(bankroll_cap_usd=cap, cash_usd=cap), args.state_file)
    Path(args.log_file).unlink(missing_ok=True)
    _log(f"Book M live state reset: bankroll_cap=${cap:,.2f}  state_file={args.state_file}")


def cmd_make_live_run(args: argparse.Namespace) -> None:
    """§4 Phase 1: tick on an interval. Dry-run (report only, no network
    writes) unless --live is passed -- see making/execution.py's module
    docstring for the full safety model."""
    import time
    from pathlib import Path

    from poly03.config import MAKING_LIVE_BANKROLL_CAP_USD
    from poly03.making.execution import run_live_tick
    from poly03.making.live_state import LiveMakingState, load_state, save_state
    from poly03.notify.telegram import TelegramReporter

    logger = logging.getLogger("poly03.making.execution")
    notifier = TelegramReporter()

    path = Path(args.state_file)
    if path.exists():
        state = load_state(args.state_file)
        notifier.log(f"resuming Book M live state from {args.state_file}: {state.n_ticks} ticks, equity=${state.equity_usd:,.2f}")
    else:
        cap = args.bankroll_cap if args.bankroll_cap is not None else MAKING_LIVE_BANKROLL_CAP_USD
        state = LiveMakingState(bankroll_cap_usd=cap, cash_usd=cap)
        save_state(state, args.state_file)
        notifier.log(f"started Book M live state: bankroll_cap=${cap:,.2f}  state_file={args.state_file}")

    mode = "LIVE -- real orders will be placed" if args.live else "DRY-RUN -- no orders placed, no network writes"
    notifier.log(f"ticking every {args.interval}s [{mode}]. Ctrl+C to stop.\n")
    notifier.flush()

    try:
        while True:
            if state.halted and not args.force:
                notifier.log("HALTED: investigate before continuing. Re-run with --force to override.")
                notifier.flush()
                break
            try:
                report = run_live_tick(
                    state,
                    max_gamma_markets=args.max_markets,
                    max_markets_quoted=args.max_quoted,
                    dry_run=not args.live,
                    decision_log_path=args.log_file,
                )
                save_state(state, args.state_file)
                notifier.log(_report_line(report, state))
            except Exception as exc:
                logger.warning("live tick failed, will retry next interval: %s", exc)
            notifier.flush()
            time.sleep(args.interval)
    except KeyboardInterrupt:
        notifier.log("\nstopping (Ctrl+C received)...")
    finally:
        save_state(state, args.state_file)
        notifier.log(
            f"\nfinal equity: ${state.equity_usd:,.2f}  open positions: {len(state.open_positions)}  "
            f"open orders: {len(state.open_orders)}  fills: {len(state.fills)}"
        )
        notifier.log("\n=== final §4 Phase 1 report ===\n")
        cmd_make_live_report(args, log=notifier.log)
        notifier.flush()


def cmd_montecarlo(args: argparse.Namespace) -> None:
    from poly03.backtest.montecarlo import simulate_equity_paths, uniform_book

    positions = uniform_book(
        args.n_positions,
        price=args.price,
        true_prob=args.true_prob,
        stake=args.stake,
        cluster_size=args.cluster_size,
    )
    result = simulate_equity_paths(
        positions,
        bankroll=args.bankroll,
        n_sims=args.n_sims,
        cluster_stress_prob=args.cluster_stress_prob,
        cluster_stress_loss_fraction=args.cluster_stress_loss_fraction,
        rng_seed=args.seed,
    )
    _log(result.summary())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="poly03", description="Polymarket 'boring edge' bot -- Phase 0 research stack")
    parser.add_argument("-v", "--verbose", action="store_true", help="enable debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    p_scan = sub.add_parser("scan", help="scan the live open-market universe for Book A candidates")
    p_scan.add_argument("--max-markets", type=int, default=200)
    p_scan.add_argument("--page-size", type=int, default=50)
    p_scan.add_argument("--target-size", type=float, default=100.0, help="intended position size in USD, for the liquidity_factor penalty")
    p_scan.set_defaults(func=cmd_scan)

    p_backtest = sub.add_parser("backtest", help="Phase 0 backtest: classifier calibration against historical resolutions")
    p_backtest.add_argument("--max-markets", type=int, default=100)
    p_backtest.add_argument("--page-size", type=int, default=20)
    p_backtest.set_defaults(func=cmd_backtest)

    p_paper = sub.add_parser("paper", help="§8 Phase 1: paper trading (no live orders, no real capital)")
    paper_sub = p_paper.add_subparsers(dest="paper_command", required=True)

    def _add_state_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--state-file", default=PAPER_STATE_FILE)
        p.add_argument("--log-file", default=PAPER_DECISION_LOG_FILE)

    p_tick = paper_sub.add_parser("tick", help="run one scan/manage/enter cycle")
    _add_state_args(p_tick)
    p_tick.add_argument("--max-markets", type=int, default=PAPER_TARGET_SCAN_MARKETS)
    p_tick.add_argument("--force", action="store_true", help="tick even if manual_review_required is set")
    p_tick.set_defaults(func=cmd_paper_tick)

    p_status = paper_sub.add_parser("status", help="show current bankroll/positions/halt state")
    _add_state_args(p_status)
    p_status.set_defaults(func=cmd_paper_status)

    p_report = paper_sub.add_parser("report", help="§7 measurement report + §8 Phase 1 gate status")
    _add_state_args(p_report)
    p_report.set_defaults(func=cmd_paper_report)

    p_reset = paper_sub.add_parser("reset", help="wipe paper state and start over")
    _add_state_args(p_reset)
    p_reset.add_argument("--bankroll", type=float, default=None)
    p_reset.set_defaults(func=cmd_paper_reset)

    p_run = paper_sub.add_parser(
        "run", help="run forever: tick on an interval, Ctrl+C to stop and print the §7 report"
    )
    _add_state_args(p_run)
    p_run.add_argument("--bankroll", type=float, default=None, help="starting bankroll, only used if state file doesn't exist yet")
    p_run.add_argument("--interval", type=int, default=1800, help="seconds between ticks (default 1800s = 30min, per §2.1 scan cadence)")
    p_run.add_argument("--max-markets", type=int, default=PAPER_TARGET_SCAN_MARKETS)
    p_run.add_argument("--force", action="store_true", help="keep running even if manual_review_required is set")
    p_run.set_defaults(func=cmd_paper_run)

    p_make = sub.add_parser("make", help="strategy_v2.md §3 Book M: reward-subsidized making (no live orders)")
    make_sub = p_make.add_subparsers(dest="make_command", required=True)

    def _add_making_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--state-file", default=MAKING_STATE_FILE)
        p.add_argument("--log-file", default=MAKING_DECISION_LOG_FILE)
        p.add_argument("--max-markets", type=int, default=GAMMA_MAX_SCAN_MARKETS, help="Gamma markets to scan per tick (Gamma 422s past ~2100, so higher has no effect)")
        p.add_argument("--max-quoted", type=int, default=MAKING_MAX_MARKETS_QUOTED)

    p_mscan = make_sub.add_parser("scan", help="§3.1: show the quotable universe (read-only)")
    p_mscan.add_argument("--max-markets", type=int, default=GAMMA_MAX_SCAN_MARKETS)
    p_mscan.add_argument("--limit", type=int, default=30)
    p_mscan.set_defaults(func=cmd_make_scan)

    p_mtick = make_sub.add_parser("tick", help="one scan/quote/score cycle")
    _add_making_args(p_mtick)
    p_mtick.set_defaults(func=cmd_make_tick)

    p_mreport = make_sub.add_parser("report", help="§4 Phase 0 reward-share measurement + gate")
    _add_making_args(p_mreport)
    p_mreport.set_defaults(func=cmd_make_report)

    p_mreset = make_sub.add_parser("reset", help="wipe Book M Phase 0 state")
    _add_making_args(p_mreset)
    p_mreset.add_argument("--bankroll", type=float, default=None)
    p_mreset.set_defaults(func=cmd_make_reset)

    p_mrun = make_sub.add_parser("run", help="§4 Phase 0: tick forever, Ctrl+C to stop and report")
    _add_making_args(p_mrun)
    p_mrun.add_argument("--bankroll", type=float, default=None)
    p_mrun.add_argument("--interval", type=int, default=1800)
    p_mrun.set_defaults(func=cmd_make_run)

    p_mlive = make_sub.add_parser(
        "live", help="§4 Phase 1: micro-live execution -- real orders only with --live, dry-run otherwise"
    )
    live_sub = p_mlive.add_subparsers(dest="live_command", required=True)

    def _add_live_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--state-file", default=MAKING_LIVE_STATE_FILE)
        p.add_argument("--log-file", default=MAKING_LIVE_DECISION_LOG_FILE)
        p.add_argument("--max-markets", type=int, default=GAMMA_MAX_SCAN_MARKETS)
        p.add_argument("--max-quoted", type=int, default=MAKING_MAX_MARKETS_QUOTED)
        p.add_argument(
            "--live", action="store_true", help="place real orders; default is dry-run (no network writes)"
        )

    p_ltick = live_sub.add_parser("tick", help="one reconcile/unwind/quote cycle")
    _add_live_args(p_ltick)
    p_ltick.add_argument("--force", action="store_true", help="tick even if state is halted")
    p_ltick.set_defaults(func=cmd_make_live_tick)

    p_lstatus = live_sub.add_parser("status", help="show live bankroll/positions/orders")
    p_lstatus.add_argument("--state-file", default=MAKING_LIVE_STATE_FILE)
    p_lstatus.set_defaults(func=cmd_make_live_status)

    p_lreport = live_sub.add_parser("report", help="§4 Phase 1 measurement: realized vs. estimated, fill rate, adverse selection, gate")
    p_lreport.add_argument("--state-file", default=MAKING_LIVE_STATE_FILE)
    p_lreport.add_argument("--log-file", default=MAKING_LIVE_DECISION_LOG_FILE)
    p_lreport.add_argument("--phase0-state-file", default=MAKING_STATE_FILE, help="Phase 0 state file to compare the realized reward rate against")
    p_lreport.set_defaults(func=cmd_make_live_report)

    p_lreward = live_sub.add_parser(
        "record-reward",
        help="manually log an observed reward payout (rewards aren't reconcilable per-order, see live_state.py)",
    )
    p_lreward.add_argument("--state-file", default=MAKING_LIVE_STATE_FILE)
    p_lreward.add_argument("--amount", type=float, required=True)
    p_lreward.add_argument("--note", type=str, default="")
    p_lreward.set_defaults(func=cmd_make_live_record_reward)

    p_lreset = live_sub.add_parser("reset", help="wipe Book M live state")
    p_lreset.add_argument("--state-file", default=MAKING_LIVE_STATE_FILE)
    p_lreset.add_argument("--log-file", default=MAKING_LIVE_DECISION_LOG_FILE)
    p_lreset.add_argument("--bankroll-cap", type=float, default=None)
    p_lreset.set_defaults(func=cmd_make_live_reset)

    p_lrun = live_sub.add_parser("run", help="tick forever on an interval, Ctrl+C to stop")
    _add_live_args(p_lrun)
    p_lrun.add_argument("--bankroll-cap", type=float, default=None)
    p_lrun.add_argument("--interval", type=int, default=1800)
    p_lrun.add_argument("--force", action="store_true", help="keep running even if state is halted")
    p_lrun.add_argument("--phase0-state-file", default=MAKING_STATE_FILE, help="Phase 0 state file to compare the realized reward rate against")
    p_lrun.set_defaults(func=cmd_make_live_run)

    p_mc = sub.add_parser("montecarlo", help="§4.2 Monte Carlo equity simulation")
    p_mc.add_argument("--n-positions", type=int, default=100)
    p_mc.add_argument("--price", type=float, default=0.91)
    p_mc.add_argument("--true-prob", type=float, default=0.95)
    p_mc.add_argument("--stake", type=float, default=500.0)
    p_mc.add_argument("--bankroll", type=float, default=100_000.0)
    p_mc.add_argument("--n-sims", type=int, default=20_000)
    p_mc.add_argument("--cluster-size", type=int, default=None, help="group positions into clusters of this size")
    p_mc.add_argument("--cluster-stress-prob", type=float, default=0.0, help="probability of a cluster-wide correlated shock")
    p_mc.add_argument("--cluster-stress-loss-fraction", type=float, default=1.0)
    p_mc.add_argument("--seed", type=int, default=None)
    p_mc.set_defaults(func=cmd_montecarlo)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.WARNING)
    args.func(args)


if __name__ == "__main__":
    sys.exit(main())
