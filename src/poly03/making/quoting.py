"""§3.2: turning a quotable market into a two-sided quote.

Three constraints have to hold at once, and they pull against each other:

1. **Reward eligibility.** An order more than `max_spread` cents from the
   midpoint scores zero, and one below `min_size` scores zero. Both are
   cliffs. An ineligible order is strictly worse than no order -- it carries
   inventory risk and earns nothing -- so this module never emits one.
2. **No self-cross.** We improve the book by a tick on each side where there
   is room, and fall back to joining the touch when there isn't. §3.1's
   two-tick minimum spread is what makes the improving case possible.
3. **Inventory.** Score is highest at the midpoint, but so is adverse
   selection, and every fill leaves us holding something. As net inventory
   grows we suppress the side that would add to it.

Collateral note: on Polymarket, resting an ask on YES is economically buying
NO, so *both* sides tie up capital -- `n * price` for the bid and
`n * (1 - price)` for the ask. `QuotePair.collateral_usd` is what a market
actually costs us to quote, and it is what the portfolio caps are applied to.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from poly03.config import (
    MAKING_INVENTORY_SKEW_SATURATION,
    MAKING_QUOTE_SAFETY_MARGIN_CENTS,
)
from poly03.making.rewards import RewardConfig, combine_sides, order_score


def round_to_tick(price: float, tick: float, *, mode: str = "nearest") -> float:
    if tick <= 0:
        return price
    scaled = price / tick
    if mode == "down":
        stepped = math.floor(scaled)
    elif mode == "up":
        stepped = math.ceil(scaled)
    else:
        stepped = round(scaled)
    # Tick sizes are 0.01/0.001, so round away float noise at that precision.
    return round(stepped * tick, 6)


@dataclass
class Quote:
    side: str          # "bid" | "ask"
    price: float
    size_shares: float

    @property
    def collateral_usd(self) -> float:
        return self.size_shares * (self.price if self.side == "bid" else 1.0 - self.price)

    def distance_cents(self, midpoint: float) -> float:
        return abs(self.price - midpoint) * 100.0


@dataclass
class QuotePair:
    market_id: str
    question: str
    token_id: str
    midpoint: float
    tick_size: float
    reward: RewardConfig
    bid: Quote | None = None
    ask: Quote | None = None
    suppressed: list[str] = None  # sides dropped, with why

    def __post_init__(self) -> None:
        if self.suppressed is None:
            self.suppressed = []

    @property
    def collateral_usd(self) -> float:
        return sum(q.collateral_usd for q in (self.bid, self.ask) if q is not None)

    @property
    def is_empty(self) -> bool:
        return self.bid is None and self.ask is None

    def qscore(self) -> float:
        """Our Q for this market under the reconstructed scoring."""
        qbid = order_score(self.bid.price, self.bid.size_shares, self.midpoint, self.reward) if self.bid else 0.0
        qask = order_score(self.ask.price, self.ask.size_shares, self.midpoint, self.reward) if self.ask else 0.0
        return combine_sides(qbid, qask, self.midpoint)


def _eligible_distance(reward: RewardConfig) -> float:
    """Furthest we can rest from the midpoint and still score, in price units,
    with a safety margin so a one-tick mid move doesn't silently drop us out
    of eligibility between scans."""
    cents = max(0.0, reward.max_spread_cents - MAKING_QUOTE_SAFETY_MARGIN_CENTS)
    return cents / 100.0


def inventory_skew(net_shares: float, inventory_cap_shares: float) -> float:
    """+1 = maximally long (stop bidding), -1 = maximally short (stop offering)."""
    if inventory_cap_shares <= 0:
        return 0.0
    raw = net_shares / (inventory_cap_shares * MAKING_INVENTORY_SKEW_SATURATION)
    return max(-1.0, min(1.0, raw))


def build_quote_pair(
    *,
    market_id: str,
    question: str,
    token_id: str,
    best_bid: float,
    best_ask: float,
    tick_size: float,
    reward: RewardConfig,
    target_size_shares: float,
    net_inventory_shares: float = 0.0,
    inventory_cap_shares: float = 0.0,
) -> QuotePair:
    """Construct the two-sided quote we would rest in this market."""
    midpoint = (best_bid + best_ask) / 2.0
    pair = QuotePair(
        market_id=market_id,
        question=question,
        token_id=token_id,
        midpoint=midpoint,
        tick_size=tick_size,
        reward=reward,
    )

    # (2) improve by a tick where there's room, else join the touch.
    bid_price = round_to_tick(best_bid + tick_size, tick_size, mode="down")
    ask_price = round_to_tick(best_ask - tick_size, tick_size, mode="up")
    if bid_price >= ask_price:
        bid_price, ask_price = best_bid, best_ask

    # (1) pull inside the reward-eligibility boundary if the book is wider
    # than the scoring window. `max`/`min` here move us *toward* the midpoint,
    # which costs us price but is the only way to score at all.
    eligible = _eligible_distance(reward)
    bid_price = round_to_tick(max(bid_price, midpoint - eligible), tick_size, mode="up")
    ask_price = round_to_tick(min(ask_price, midpoint + eligible), tick_size, mode="down")
    if bid_price >= ask_price:
        pair.suppressed.append("reward_window_narrower_than_one_tick")
        return pair

    # (3) inventory skew: scale the adding side down as we accumulate.
    skew = inventory_skew(net_inventory_shares, inventory_cap_shares)
    bid_size = target_size_shares * (1.0 - max(0.0, skew))
    ask_size = target_size_shares * (1.0 - max(0.0, -skew))

    # An order below min_size scores nothing, so post it at full size or not
    # at all -- never a stub that carries risk without earning.
    if bid_size >= reward.min_size:
        pair.bid = Quote(side="bid", price=bid_price, size_shares=bid_size)
    else:
        pair.suppressed.append("bid_below_reward_min_size")
    if ask_size >= reward.min_size:
        pair.ask = Quote(side="ask", price=ask_price, size_shares=ask_size)
    else:
        pair.suppressed.append("ask_below_reward_min_size")

    return pair


def needs_requote(quoted_midpoint: float, current_midpoint: float, threshold_cents: float) -> bool:
    """§3.2: cancel/replace once the midpoint has moved enough that our resting
    quote is stale. Stale quotes are the adverse-selection channel (§3.4)."""
    return abs(current_midpoint - quoted_midpoint) * 100.0 >= threshold_cents
