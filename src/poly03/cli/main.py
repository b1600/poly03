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
    print(msg)
    append_log(msg)


def cmd_scan(args: argparse.Namespace) -> None:
    """Apply exclusion filters + classifier + ROC scoring to the current
    open-market universe and print surviving Book A candidates.

    q (estimated true probability) has no model wired up here -- it's a
    placeholder floor of `price + MIN_MARGIN_PP`, which by construction
    just barely clears the margin gate. Replace with a real probability
    estimate before using edge_score for anything beyond a rough scan.
    """
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
