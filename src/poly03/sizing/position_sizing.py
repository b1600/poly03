"""§4.1: fixed-fractional sizing with a hard Kelly cap.

Full Kelly is explicitly rejected in the doc -- at p=0.92, q=0.99 it wants
87.5% of bankroll, which is just an artifact of pretending we know q to two
decimals. We use 1/10 Kelly as a ceiling, not a target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from poly03.classifier.taxonomy import Tier
from poly03.config import (
    BASE_FRACTION,
    KELLY_FRACTION_CAP,
    LIQUIDITY_SIZE_FRACTION_CAP,
    MAX_BOOK_A_DEPLOYED_FRACTION,
    MAX_BOOK_B_DEPLOYED_FRACTION,
    MAX_SINGLE_POSITION_FRACTION,
    MIN_CASH_RESERVE_FRACTION,
    TIER_SIZE_MULTIPLIER,
)


def kelly_fraction(q: float, p: float) -> float:
    """Full-Kelly fraction of bankroll for a binary bet: buy at price p
    (payout $1 if right, $0 if wrong), true win probability q. Net odds
    b = (1-p)/p; f* = q - (1-q)/b. Clamped to [0, 1] -- negative means no
    edge (q <= p), which shouldn't reach this function if §2.3's margin
    gate was applied first, but we don't assume the caller did that."""
    if not 0 < p < 1:
        raise ValueError(f"price must be in (0, 1), got {p}")
    b = (1 - p) / p
    f = q - (1 - q) / b
    return max(0.0, min(1.0, f))


@dataclass
class SizingInputs:
    bankroll: float
    tier: Tier
    maker_price: float
    estimated_true_probability: float
    visible_book_depth_usd: float
    max_position_usd: float | None = None  # override; defaults to MAX_SINGLE_POSITION_FRACTION * bankroll


@dataclass
class SizingResult:
    stake_usd: float
    binding_constraint: str
    kelly_fraction_raw: float
    components: dict[str, float] = field(default_factory=dict)


def compute_stake(inputs: SizingInputs) -> SizingResult:
    tier_multiplier = TIER_SIZE_MULTIPLIER[int(inputs.tier)]
    kf = kelly_fraction(inputs.estimated_true_probability, inputs.maker_price)

    max_position = (
        inputs.max_position_usd
        if inputs.max_position_usd is not None
        else MAX_SINGLE_POSITION_FRACTION * inputs.bankroll
    )

    components = {
        "base_fraction": BASE_FRACTION * inputs.bankroll * tier_multiplier,
        "kelly_capped": kf * KELLY_FRACTION_CAP * inputs.bankroll,
        "max_position": max_position,
        "depth_cap": LIQUIDITY_SIZE_FRACTION_CAP * inputs.visible_book_depth_usd,
    }
    binding = min(components, key=components.get)
    stake = max(0.0, components[binding])

    return SizingResult(stake_usd=stake, binding_constraint=binding, kelly_fraction_raw=kf, components=components)


@dataclass
class PortfolioCapCheck:
    ok: bool
    reasons: list[str] = field(default_factory=list)


def check_portfolio_caps(
    *,
    book: str,
    bankroll: float,
    cash_available: float,
    book_a_deployed_usd: float,
    book_b_deployed_usd: float,
    proposed_stake_usd: float,
) -> PortfolioCapCheck:
    """§4.1 portfolio-level caps: max Book A/B deployment, min cash reserve.
    Checked in addition to (not instead of) the per-position caps in
    compute_stake()."""
    reasons = []

    if cash_available - proposed_stake_usd < MIN_CASH_RESERVE_FRACTION * bankroll:
        reasons.append(f"would breach min cash reserve ({MIN_CASH_RESERVE_FRACTION:.0%} of bankroll)")

    if book.upper() == "A":
        if book_a_deployed_usd + proposed_stake_usd > MAX_BOOK_A_DEPLOYED_FRACTION * bankroll:
            reasons.append(f"would breach max Book A deployment ({MAX_BOOK_A_DEPLOYED_FRACTION:.0%} of bankroll)")
    elif book.upper() == "B":
        if book_b_deployed_usd + proposed_stake_usd > MAX_BOOK_B_DEPLOYED_FRACTION * bankroll:
            reasons.append(f"would breach max Book B deployment ({MAX_BOOK_B_DEPLOYED_FRACTION:.0%} of bankroll)")

    return PortfolioCapCheck(ok=len(reasons) == 0, reasons=reasons)
