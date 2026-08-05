"""§3.1 tiered question taxonomy."""

from __future__ import annotations

from enum import IntEnum

from poly03.config import TIER_CONFIDENCE_MULTIPLIER


class Tier(IntEnum):
    """Lower number = more structurally certain. Tier 4 is excluded from
    Book A entirely -- it exists so 'no confident tier' has somewhere to go
    other than silently defaulting to a tradeable one."""

    TIER_1 = 1  # physical / structural impossibility
    TIER_2 = 2  # status-quo inertia with a hard clock
    TIER_3 = 3  # base-rate favorite, no current catalyst
    TIER_4 = 4  # requires a forecast -- no edge, excluded


TIER_LABELS: dict[Tier, str] = {
    Tier.TIER_1: "physical/structural impossibility",
    Tier.TIER_2: "status-quo inertia with a hard clock",
    Tier.TIER_3: "base-rate favorite",
    Tier.TIER_4: "requires a forecast (excluded)",
}


def confidence_multiplier(tier: Tier) -> float:
    return TIER_CONFIDENCE_MULTIPLIER[int(tier)]
