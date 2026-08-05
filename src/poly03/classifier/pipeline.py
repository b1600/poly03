"""Composes the mechanical rules engine with the LLM veto layer and logs
every classification with its inputs (§3.2), so the classifier can later be
back-tested independently of the trading strategy itself.
"""

from __future__ import annotations

import logging

from poly03.classifier.llm_veto import LLMClassifierVeto, NoOpVeto, apply_veto
from poly03.classifier.rules import Classification, classify_market_rules
from poly03.data.models import Market

logger = logging.getLogger("poly03.classifier")


def classify_market(market: Market, veto: LLMClassifierVeto | None = None) -> Classification:
    veto = veto or NoOpVeto()
    rules_result = classify_market_rules(market)
    veto_result = veto.review(market, rules_result)
    final = apply_veto(rules_result, veto_result)

    logger.info(
        "classification",
        extra={
            "market_id": final.market_id,
            "tier": final.tier.value,
            "confidence_multiplier": final.confidence_multiplier,
            "conservative_fallback": final.conservative_fallback,
            "evidence": final.evidence,
            "inputs": final.inputs,
        },
    )
    return final
