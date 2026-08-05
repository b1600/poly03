"""§3.2's LLM-assisted layer: "a veto and a tier-capper, not a promoter."

An LLM saying "this is obvious" must never raise a tier the rules engine
(rules.py) assigned; an LLM flagging ambiguity always lowers one (or sends
it straight to Tier 4). No LLM is wired up by default -- NoOpVeto passes
the rules-engine tier through unchanged, which keeps the pipeline usable
for backtesting before any LLM integration exists. Plug a real
implementation in by satisfying LLMClassifierVeto and requiring structured
output with an explicit confidence and a cited clause from the rules text,
per the doc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from poly03.classifier.rules import Classification
from poly03.classifier.taxonomy import Tier, confidence_multiplier
from poly03.data.models import Market


@dataclass
class VetoResult:
    downgraded: bool
    tier: Tier  # the tier this veto is willing to allow, at most
    confidence: float  # LLM's stated confidence in its own read, 0-1
    cited_clause: str | None
    rationale: str


class LLMClassifierVeto(Protocol):
    def review(self, market: Market, classification: Classification) -> VetoResult: ...


class NoOpVeto:
    def review(self, market: Market, classification: Classification) -> VetoResult:
        return VetoResult(
            downgraded=False,
            tier=classification.tier,
            confidence=1.0,
            cited_clause=None,
            rationale="no LLM veto configured; rules-engine tier passed through unchanged",
        )


def apply_veto(classification: Classification, veto: VetoResult) -> Classification:
    """Merge a veto result, enforcing the never-promote invariant even if a
    misbehaving veto implementation tries to raise the tier."""
    final_tier = Tier(max(int(classification.tier), int(veto.tier)))
    evidence = list(classification.evidence)
    if final_tier != classification.tier:
        evidence.append(
            f"LLM veto downgraded tier {classification.tier.value} -> {final_tier.value}: "
            f"{veto.rationale} (cited: {veto.cited_clause!r}, confidence={veto.confidence:.2f})"
        )
    return Classification(
        market_id=classification.market_id,
        tier=final_tier,
        confidence_multiplier=confidence_multiplier(final_tier),
        evidence=evidence,
        conservative_fallback=classification.conservative_fallback,
        inputs=classification.inputs,
    )
