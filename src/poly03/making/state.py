"""Book M state (strategy_v2.md §4, Phase 0).

Deliberately smaller than `paper/state.py`, because Phase 0 for Book M holds
no positions. It does not simulate fills at all -- v1's paper engine assumed
every resting order filled in full, which flattered it, and the honest version
of that assumption for a making book is that we cannot know. Fills, the
realized maker fee, and adverse selection are all Phase 1 (micro-live)
measurements per §4.

So what accumulates here is an *observation series*: at each tick, what we
would have quoted and what share of each reward pool that quote would have
scored. Integrating that series over the observation window is the deliverable
§4 asks for -- a defensible number for our share of the pool.

Per-tick summaries live in the JSON state file; the full per-market detail is
appended to the JSONL log, same split as `paper/state.py`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from poly03.config import MAKING_STATE_FILE, PAPER_STARTING_BANKROLL


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class MarketObservation:
    """One market, one tick: the quote we'd have rested and what it'd score."""

    market_id: str
    condition_id: str
    question: str
    midpoint: float
    best_bid: float
    best_ask: float
    tick_size: float
    reward_daily_rate: float
    reward_min_size: float
    reward_max_spread_cents: float
    our_qscore: float
    competing_qscore: float
    share_fraction: float
    est_reward_usd_per_day: float
    collateral_usd: float
    identified: bool = True
    raw_share_fraction: float = 0.0
    bid_price: float | None = None
    bid_size: float | None = None
    ask_price: float | None = None
    ask_size: float | None = None
    suppressed: list[str] = field(default_factory=list)


@dataclass
class MakingTickSummary:
    timestamp: str
    gamma_scanned: int
    reward_eligible: int
    quotable: int
    quoted: int
    total_collateral_usd: float
    total_est_reward_usd_per_day: float
    pool_usd_per_day_in_quoted_markets: float
    # Portion of the estimate coming from markets with no competing depth
    # inside the scoring window -- unidentified, so quarantined in reports.
    unidentified_est_reward_usd_per_day: float = 0.0
    unidentified_quoted: int = 0
    rejections: dict[str, int] = field(default_factory=dict)

    @property
    def identified_est_reward_usd_per_day(self) -> float:
        """The defensible part of the estimate: markets where we actually
        observed competition to measure our share against."""
        return self.total_est_reward_usd_per_day - self.unidentified_est_reward_usd_per_day

    @property
    def daily_yield_on_collateral(self) -> float:
        if self.total_collateral_usd <= 0:
            return 0.0
        return self.total_est_reward_usd_per_day / self.total_collateral_usd


@dataclass
class MakingState:
    bankroll: float = PAPER_STARTING_BANKROLL
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    n_ticks: int = 0
    ticks: list[MakingTickSummary] = field(default_factory=list)
    # market_id -> cumulative estimated reward-USD-days, for per-market ranking
    market_reward_accrual: dict[str, float] = field(default_factory=dict)

    @property
    def observation_days(self) -> float:
        if len(self.ticks) < 2:
            return 0.0
        start = datetime.fromisoformat(self.ticks[0].timestamp)
        end = datetime.fromisoformat(self.ticks[-1].timestamp)
        return max(0.0, (end - start).total_seconds() / 86400.0)

    def record(self, summary: MakingTickSummary, observations: list[MarketObservation]) -> None:
        self.n_ticks += 1
        self.ticks.append(summary)
        for obs in observations:
            prior = self.market_reward_accrual.get(obs.market_id, 0.0)
            self.market_reward_accrual[obs.market_id] = prior + obs.est_reward_usd_per_day


def load_state(path: str | Path = MAKING_STATE_FILE) -> MakingState:
    p = Path(path)
    if not p.exists():
        return MakingState()
    raw = json.loads(p.read_text())
    ticks = [MakingTickSummary(**t) for t in raw.pop("ticks", [])]
    state = MakingState(**raw)
    state.ticks = ticks
    return state


def save_state(state: MakingState, path: str | Path = MAKING_STATE_FILE) -> None:
    state.updated_at = _now_iso()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(asdict(state), indent=2, default=str))


def log_observation(event: dict[str, Any], path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps({"timestamp": _now_iso(), **event}, default=str) + "\n")
