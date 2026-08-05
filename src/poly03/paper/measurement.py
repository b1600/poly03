"""§7 measurement, computed from paper-trading state.

Calibration by tier is "the single most important number" per the doc --
everything else here exists to answer *why* the strategy is or isn't
working, calibration exists to answer *whether* the classifier can be
trusted at all.

Only positions that reached actual resolution (`resolved_win` /
`resolved_loss`) count toward calibration/Brier/ROC -- early exits
(falsification, dispute, recycling) didn't get a chance to prove the
classifier right or wrong, so mixing them in would understate the sample
size problem rather than solve it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from poly03.config import PAPER_GATE_CALIBRATION_TOLERANCE_PP, PAPER_GATE_MIN_RESOLUTIONS
from poly03.paper.state import PaperPosition, PaperState

_RESOLVED_STATUSES = ("resolved_win", "resolved_loss")


def _resolved(state: PaperState) -> list[PaperPosition]:
    return [p for p in state.closed_positions if p.status in _RESOLVED_STATUSES]


@dataclass
class CalibrationBucket:
    tier: int
    n: int
    mean_entry_price: float
    realized_win_rate: float
    brier_score: float


def calibration_by_tier(state: PaperState) -> list[CalibrationBucket]:
    buckets: dict[int, list[PaperPosition]] = {}
    for p in _resolved(state):
        buckets.setdefault(p.tier, []).append(p)
    out = []
    for tier, items in sorted(buckets.items()):
        n = len(items)
        mean_p = sum(i.entry_price for i in items) / n
        win_rate = sum(1 for i in items if i.status == "resolved_win") / n
        brier = sum((i.entry_price - (1.0 if i.status == "resolved_win" else 0.0)) ** 2 for i in items) / n
        out.append(CalibrationBucket(tier=tier, n=n, mean_entry_price=mean_p, realized_win_rate=win_rate, brier_score=brier))
    return out


def overall_brier(state: PaperState) -> float | None:
    items = _resolved(state)
    if not items:
        return None
    return sum((i.entry_price - (1.0 if i.status == "resolved_win" else 0.0)) ** 2 for i in items) / len(items)


def realized_vs_modeled_roc(state: PaperState) -> tuple[float, float] | None:
    """(mean modeled annualized ROC at entry, mean realized annualized ROC).
    The gap is fees + slippage + lockup we underestimated -- except paper
    trading can't see fees/slippage at all (§8: that's what Phase 2 is
    for), so a nonzero gap here reflects timing/classifier error only."""
    items = _resolved(state)
    if not items:
        return None
    modeled = sum(i.modeled_annualized_roc for i in items) / len(items)
    realized = []
    for i in items:
        gross = (i.realized_pnl or 0.0) / i.stake_usd if i.stake_usd else 0.0
        days = i.realized_days_held or 0.0
        if days > 0:
            realized.append((1 + gross) ** (365.0 / days) - 1)
        else:
            realized.append(gross)
    return modeled, sum(realized) / len(realized)


@dataclass
class DrawdownStats:
    max_drawdown_fraction: float
    max_drawdown_usd: float
    current_drawdown_fraction: float
    time_to_recovery_days: float | None  # None if never recovered (or no drawdown yet)


def drawdown_stats(state: PaperState) -> DrawdownStats:
    closed = sorted((p for p in state.closed_positions if p.closed_at), key=lambda p: p.closed_at)
    equity = state.bankroll
    running_max = equity
    max_dd_frac = 0.0
    max_dd_usd = 0.0
    peak_ts: datetime | None = None
    trough_ts: datetime | None = None
    recovery_days: float | None = None
    in_drawdown = False

    for p in closed:
        equity += p.realized_pnl or 0.0
        ts = datetime.fromisoformat(p.closed_at)
        if equity >= running_max:
            if in_drawdown and peak_ts is not None:
                recovery_days = (ts - peak_ts).total_seconds() / 86400.0
            running_max = equity
            in_drawdown = False
        else:
            in_drawdown = True
            dd_usd = running_max - equity
            dd_frac = dd_usd / running_max if running_max > 0 else 0.0
            if dd_frac > max_dd_frac:
                max_dd_frac = dd_frac
                max_dd_usd = dd_usd
                peak_ts = peak_ts or ts
                trough_ts = ts

    current_dd = (running_max - equity) / running_max if running_max > 0 else 0.0
    return DrawdownStats(
        max_drawdown_fraction=max_dd_frac,
        max_drawdown_usd=max_dd_usd,
        current_drawdown_fraction=current_dd,
        time_to_recovery_days=recovery_days if not in_drawdown else None,
    )


def pnl_by_tier(state: PaperState) -> dict[int, float]:
    out: dict[int, float] = {}
    for p in state.closed_positions:
        out[p.tier] = out.get(p.tier, 0.0) + (p.realized_pnl or 0.0)
    return out


def pnl_by_cluster_entity(state: PaperState) -> dict[str, float]:
    out: dict[str, float] = {}
    for p in state.closed_positions:
        entity = p.cluster_tags.get("entity", "unknown")
        out[entity] = out.get(entity, 0.0) + (p.realized_pnl or 0.0)
    return out


def exit_reason_counts(state: PaperState) -> dict[str, int]:
    out: dict[str, int] = {}
    for p in state.closed_positions:
        reason = p.close_reason or "unknown"
        out[reason] = out.get(reason, 0) + 1
    return out


@dataclass
class Phase1GateStatus:
    n_resolutions: int
    min_required: int
    tier1_misses: int
    calibration_ok: bool
    calibration_detail: list[str] = field(default_factory=list)
    ready_to_advance: bool = False

    def summary(self) -> str:
        lines = [
            f"resolutions: {self.n_resolutions}/{self.min_required}"
            + (" [OK]" if self.n_resolutions >= self.min_required else " [NOT ENOUGH]"),
            f"Tier 1 misses: {self.tier1_misses}" + (" [OK]" if self.tier1_misses == 0 else " [FAIL -- full stop required]"),
            f"calibration within tolerance: {'yes' if self.calibration_ok else 'no'}",
        ]
        lines.extend(f"  {d}" for d in self.calibration_detail)
        lines.append(f"\nready to advance to Phase 2 (micro-live): {'YES' if self.ready_to_advance else 'NO'}")
        return "\n".join(lines)


def phase1_gate_status(state: PaperState) -> Phase1GateStatus:
    resolved = _resolved(state)
    n = len(resolved)
    tier1_misses = sum(1 for p in resolved if p.tier == 1 and p.status == "resolved_loss")

    buckets = calibration_by_tier(state)
    calibration_ok = True
    detail = []
    for b in buckets:
        gap = abs(b.realized_win_rate - b.mean_entry_price)
        ok = gap <= PAPER_GATE_CALIBRATION_TOLERANCE_PP
        calibration_ok = calibration_ok and ok
        detail.append(
            f"Tier {b.tier}: n={b.n} mean_entry={b.mean_entry_price:.3f} "
            f"realized_win_rate={b.realized_win_rate:.3f} gap={gap:.3f} "
            f"({'ok' if ok else 'OUT OF TOLERANCE'})"
        )

    ready = n >= PAPER_GATE_MIN_RESOLUTIONS and tier1_misses == 0 and calibration_ok
    return Phase1GateStatus(
        n_resolutions=n,
        min_required=PAPER_GATE_MIN_RESOLUTIONS,
        tier1_misses=tier1_misses,
        calibration_ok=calibration_ok,
        calibration_detail=detail,
        ready_to_advance=ready,
    )
