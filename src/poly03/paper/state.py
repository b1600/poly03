"""Phase 1 (§8) paper-trading state: positions + bankroll, persisted to a
JSON file between `poly03 paper tick` invocations, plus an append-only
JSONL decision log (§5.3: "structured logging of every decision, including
*rejections* and why -- the rejection log is the training data for
classifier v2", and §7's counterfactual log).

This is deliberately a flat JSON file, not a database -- Phase 1 runs at a
15-30 minute scan cadence per strategy_v1.md §2.1, not high frequency, and
a single operator-run process is the expected deployment shape at this
phase. Revisit if Phase 2 needs concurrent writers.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from poly03.cluster.tagging import ClusterTags
from poly03.config import PAPER_DECISION_LOG_FILE, PAPER_STARTING_BANKROLL, PAPER_STATE_FILE

PositionStatus = Literal["open", "resolved_win", "resolved_loss", "exited_early"]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class PaperPosition:
    id: str
    market_id: str
    question: str
    token_id: str
    outcome: str
    side_index: int
    tier: int
    entry_price: float
    stake_usd: float
    shares: float
    opened_at: str
    end_date: str | None
    days_to_resolution_at_entry: float
    modeled_annualized_roc: float
    cluster_tags: dict[str, Any]
    book: str = "A"
    status: PositionStatus = "open"
    close_reason: str | None = None
    closed_at: str | None = None
    close_price: float | None = None
    realized_pnl: float | None = None
    realized_days_held: float | None = None

    @property
    def cluster(self) -> ClusterTags:
        t = self.cluster_tags
        return ClusterTags(
            market_id=self.market_id,
            entity=t["entity"],
            themes=tuple(t.get("themes", ())),
            geography=t.get("geography"),
            resolution_source=t["resolution_source"],
            date_bucket=t.get("date_bucket"),
        )


@dataclass
class PaperState:
    bankroll: float = PAPER_STARTING_BANKROLL
    cash: float = PAPER_STARTING_BANKROLL
    high_water_mark: float = PAPER_STARTING_BANKROLL
    positions: list[PaperPosition] = field(default_factory=list)
    halted: bool = False
    halt_reasons: list[str] = field(default_factory=list)
    manual_review_required: bool = False
    created_at: str = field(default_factory=_now_iso)
    updated_at: str = field(default_factory=_now_iso)
    n_ticks: int = 0

    @property
    def open_positions(self) -> list[PaperPosition]:
        return [p for p in self.positions if p.status == "open"]

    @property
    def closed_positions(self) -> list[PaperPosition]:
        return [p for p in self.positions if p.status != "open"]

    @property
    def deployed_usd(self) -> float:
        return sum(p.stake_usd for p in self.open_positions)

    @property
    def book_a_deployed_usd(self) -> float:
        return sum(p.stake_usd for p in self.open_positions if p.book == "A")

    @property
    def book_b_deployed_usd(self) -> float:
        return sum(p.stake_usd for p in self.open_positions if p.book == "B")

    @property
    def equity(self) -> float:
        """Mark-to-cost equity: cash + stake basis of open positions. Open
        positions aren't marked to current book price -- realized P&L only
        counts on close, consistent with §6.1 hold-to-resolution as the
        default and keeping the equity curve driven by settled outcomes,
        not paper mark-to-market noise."""
        return self.cash + self.deployed_usd

    def new_position(
        self,
        *,
        market_id: str,
        question: str,
        token_id: str,
        outcome: str,
        side_index: int,
        tier: int,
        entry_price: float,
        stake_usd: float,
        end_date: str | None,
        days_to_resolution_at_entry: float,
        modeled_annualized_roc: float,
        cluster_tags: ClusterTags,
        book: str = "A",
    ) -> PaperPosition:
        pos = PaperPosition(
            id=str(uuid.uuid4()),
            market_id=market_id,
            question=question,
            token_id=token_id,
            outcome=outcome,
            side_index=side_index,
            tier=tier,
            entry_price=entry_price,
            stake_usd=stake_usd,
            shares=stake_usd / entry_price if entry_price > 0 else 0.0,
            opened_at=_now_iso(),
            end_date=end_date,
            days_to_resolution_at_entry=days_to_resolution_at_entry,
            modeled_annualized_roc=modeled_annualized_roc,
            cluster_tags={
                "entity": cluster_tags.entity,
                "themes": list(cluster_tags.themes),
                "geography": cluster_tags.geography,
                "resolution_source": cluster_tags.resolution_source,
                "date_bucket": cluster_tags.date_bucket,
            },
            book=book,
        )
        self.positions.append(pos)
        self.cash -= stake_usd
        return pos

    def close_position(self, pos: PaperPosition, *, status: PositionStatus, reason: str, close_price: float) -> None:
        pos.status = status
        pos.close_reason = reason
        pos.closed_at = _now_iso()
        pos.close_price = close_price
        pos.realized_pnl = pos.shares * close_price - pos.stake_usd
        opened = datetime.fromisoformat(pos.opened_at)
        pos.realized_days_held = max(0.0, (datetime.now(timezone.utc) - opened).total_seconds() / 86400.0)
        self.cash += pos.stake_usd + pos.realized_pnl
        self.high_water_mark = max(self.high_water_mark, self.equity)


def _position_from_dict(d: dict[str, Any]) -> PaperPosition:
    return PaperPosition(**d)


def load_state(path: str | Path = PAPER_STATE_FILE) -> PaperState:
    p = Path(path)
    if not p.exists():
        return PaperState()
    raw = json.loads(p.read_text())
    positions = [_position_from_dict(pd) for pd in raw.pop("positions", [])]
    state = PaperState(**raw)
    state.positions = positions
    return state


def save_state(state: PaperState, path: str | Path = PAPER_STATE_FILE) -> None:
    state.updated_at = _now_iso()
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(state)
    p.write_text(json.dumps(payload, indent=2, default=str))


def log_decision(event: dict[str, Any], path: str | Path = PAPER_DECISION_LOG_FILE) -> None:
    """Append one structured decision (entry, exit, or rejection) to the
    JSONL log -- §5.3 / §7's counterfactual log."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    event = {"timestamp": _now_iso(), **event}
    with p.open("a") as f:
        f.write(json.dumps(event, default=str) + "\n")


def iter_decisions(path: str | Path = PAPER_DECISION_LOG_FILE):
    p = Path(path)
    if not p.exists():
        return
    with p.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
