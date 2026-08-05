"""§3.2 mechanical (rules/keyword/template) classification layer.

This is deliberately coarse. strategy_v1.md is explicit that the failure
mode to fear is a Tier-3 market scored as Tier-1, so every pattern here is
written to be conservative: when a question doesn't cleanly match a
tier's signature, it falls through to Tier 4 (excluded) rather than being
guessed into something tradeable. The LLM layer in llm_veto.py sits on top
of this and may only lower a tier or veto entirely -- it never promotes,
so this module's Tier 1/2 calls are the ceiling for the whole pipeline.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from poly03.classifier.taxonomy import Tier, confidence_multiplier
from poly03.config import TIER2_MAX_HORIZON_DAYS
from poly03.data.models import Market


@dataclass
class Classification:
    market_id: str
    tier: Tier
    confidence_multiplier: float
    evidence: list[str] = field(default_factory=list)
    conservative_fallback: bool = False  # True => landed on Tier 4 by default, not by a positive Tier-4 match

    # §3.2: every auto-classification is logged with its inputs
    inputs: dict = field(default_factory=dict)


def _text_of(market: Market) -> str:
    return f"{market.question}\n{market.description}".lower()


# --- Tier 4: requires a forecast -- checked first, since it's the hard veto ------

_FORECAST_PATTERNS = (
    (re.compile(r"\bvs\.?\b", re.I), "head-to-head matchup ('vs')"),
    (re.compile(r"\bexact score\b", re.I), "exact-score sports market"),
    (re.compile(r"\b(win|wins|winner of)\b.*\b(game|match|series|tournament|championship|cup|open|masters)\b", re.I), "sports outcome"),
    (re.compile(r"\bwin(s)? the .*(nomination|election|primary|presidency)\b", re.I), "electoral outcome"),
    (re.compile(r"\bapproval rating\b", re.I), "approval-rating forecast"),
    (re.compile(r"\bpolling\b", re.I), "polling-based forecast"),
    (re.compile(r"\breach(es)?\s*\$", re.I), "price-level forecast"),
    (re.compile(r"\bclose (above|below)\b", re.I), "price-level forecast"),
    (re.compile(r"\bprice of\b.*\b(be|exceed|reach)\b", re.I), "price-level forecast"),
    (re.compile(r"\bunemployment rate\b", re.I), "macro data point"),
    (re.compile(r"\b(inflation|cpi|gdp)\b", re.I), "macro data point"),
    (re.compile(r"\binterest rates?\b.*\b(cut|hike|decrease|increase|raise|lower)\b", re.I), "rate-decision forecast"),
    (re.compile(r"\bearnings\b", re.I), "earnings forecast"),
    (re.compile(r"\bbest ai model\b", re.I), "benchmark forecast"),
)

# --- Tier 1: physical / structural impossibility ---------------------------------

_TIER1_PATTERNS = (
    re.compile(r"\brequires? a (constitutional amendment|two-thirds vote|supermajority)\b", re.I),
    re.compile(r"\bhas not (yet )?(been )?(nominated|appointed|scheduled|announced|filed)\b", re.I),
    re.compile(r"\bno .*(has been|is) scheduled\b", re.I),
    re.compile(r"\bbefore .*(can|could) (occur|happen|take place)\b", re.I),
    re.compile(r"\bprocess has not (started|begun)\b", re.I),
    re.compile(r"\bmandatory (lead time|waiting period)\b", re.I),
    re.compile(r"\bconstitutionally (required|mandated)\b", re.I),
)

# Minimum real-world lead time (days) for named structural processes -- if the
# question's remaining horizon is shorter than this, completion is a physical
# impossibility regardless of intent.
_STRUCTURAL_PROCESS_MIN_DAYS: dict[str, int] = {
    "constitutional amendment": 365,
    "impeachment and removal": 45,
    "treaty ratification": 60,
    "supreme court ruling": 90,
    "presidential election": 300,
}

# --- Tier 2: status-quo inertia with a hard clock --------------------------------

_TIER2_EVENT_PATTERNS = (
    re.compile(r"\bresigns?\b", re.I),
    re.compile(r"\bsteps? down\b", re.I),
    re.compile(r"\bremoved from power\b", re.I),
    re.compile(r"\bregime (falls?|collapses?)\b", re.I),
    re.compile(r"\btreaty (is )?signed\b", re.I),
    re.compile(r"\bceasefire (is )?signed\b", re.I),
    re.compile(r"\bdissolved\b", re.I),
    re.compile(r"\bmartial law\b", re.I),
)

_LIVE_PROCESS_PATTERNS = (
    re.compile(r"\bin talks\b", re.I),
    re.compile(r"\bnegotiations? underway\b", re.I),
    re.compile(r"\bexpected to\b", re.I),
    re.compile(r"\bannounced (his|her|their) intention\b", re.I),
    re.compile(r"\bscheduled to (sign|resign|step down)\b", re.I),
    re.compile(r"\bvote (is )?scheduled\b", re.I),
)

# --- Tier 3: base-rate favorite, no current catalyst -----------------------------

_TIER3_PATTERNS = (
    re.compile(r"\bwar between\b", re.I),
    re.compile(r"\bnuclear\b", re.I),
    re.compile(r"\binvades?\b", re.I),
    re.compile(r"\bdeclares? war\b", re.I),
    re.compile(r"\bassassinat", re.I),
    re.compile(r"\bcoup\b", re.I),
    re.compile(r"\breturns? (before|by|in)\b", re.I),  # e.g. "Jesus Christ returns before 2027"
)


def _match_any(patterns, text: str) -> str | None:
    for p in patterns:
        if p.search(text):
            return p.pattern
    return None


def classify_market_rules(market: Market) -> Classification:
    text = _text_of(market)
    days = market.days_to_resolution

    inputs = {
        "question": market.question,
        "description_snippet": market.description[:500],
        "days_to_resolution": days,
    }

    # Tier 4 hard veto first -- forecast-shaped questions are excluded no
    # matter what else matches.
    hit = _match_any([p for p, _ in _FORECAST_PATTERNS], text)
    if hit:
        label = next(lbl for p, lbl in _FORECAST_PATTERNS if p.pattern == hit)
        return Classification(
            market_id=market.id,
            tier=Tier.TIER_4,
            confidence_multiplier=confidence_multiplier(Tier.TIER_4),
            evidence=[f"forecast pattern matched: {label} ({hit!r})"],
            conservative_fallback=False,
            inputs=inputs,
        )

    # Tier 1: structural impossibility via explicit phrasing ...
    hit = _match_any(_TIER1_PATTERNS, text)
    if hit:
        return Classification(
            market_id=market.id,
            tier=Tier.TIER_1,
            confidence_multiplier=confidence_multiplier(Tier.TIER_1),
            evidence=[f"tier-1 structural phrase matched: {hit!r}"],
            inputs=inputs,
        )

    # ... or via named-process lead-time math: if the remaining horizon is
    # shorter than the process's known minimum duration, the outcome is
    # structurally impossible, which is the strongest form of "obvious."
    if days is not None:
        for process, min_days in _STRUCTURAL_PROCESS_MIN_DAYS.items():
            if process in text and days < min_days:
                return Classification(
                    market_id=market.id,
                    tier=Tier.TIER_1,
                    confidence_multiplier=confidence_multiplier(Tier.TIER_1),
                    evidence=[
                        f"named process {process!r} requires >= {min_days}d "
                        f"but only {days:.0f}d remain -- structurally impossible"
                    ],
                    inputs=inputs,
                )

    # Tier 2: status-quo inertia, hard clock, no visible live process.
    hit = _match_any(_TIER2_EVENT_PATTERNS, text)
    if hit:
        live_hit = _match_any(_LIVE_PROCESS_PATTERNS, text)
        if live_hit:
            return Classification(
                market_id=market.id,
                tier=Tier.TIER_4,
                confidence_multiplier=confidence_multiplier(Tier.TIER_4),
                evidence=[
                    f"tier-2 event pattern {hit!r} matched but a live process "
                    f"is underway ({live_hit!r}) -- not inertia, a forecast"
                ],
                inputs=inputs,
            )
        if days is not None and days <= TIER2_MAX_HORIZON_DAYS:
            return Classification(
                market_id=market.id,
                tier=Tier.TIER_2,
                confidence_multiplier=confidence_multiplier(Tier.TIER_2),
                evidence=[f"tier-2 event pattern matched: {hit!r}, {days:.0f}d <= {TIER2_MAX_HORIZON_DAYS}d clock"],
                inputs=inputs,
            )
        # same event pattern but no hard clock -> falls back to base-rate framing
        return Classification(
            market_id=market.id,
            tier=Tier.TIER_3,
            confidence_multiplier=confidence_multiplier(Tier.TIER_3),
            evidence=[f"tier-2 event pattern {hit!r} matched but window is not short (or unknown) -> base rate only"],
            inputs=inputs,
        )

    # Tier 3: recognizable "essentially never happens" shape.
    hit = _match_any(_TIER3_PATTERNS, text)
    if hit:
        return Classification(
            market_id=market.id,
            tier=Tier.TIER_3,
            confidence_multiplier=confidence_multiplier(Tier.TIER_3),
            evidence=[f"tier-3 base-rate pattern matched: {hit!r}"],
            inputs=inputs,
        )

    # Nothing matched confidently -- conservative default per §3.1.
    return Classification(
        market_id=market.id,
        tier=Tier.TIER_4,
        confidence_multiplier=confidence_multiplier(Tier.TIER_4),
        evidence=["no tier pattern matched"],
        conservative_fallback=True,
        inputs=inputs,
    )
