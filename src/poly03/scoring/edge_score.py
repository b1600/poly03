"""§2.3: Book A candidate scoring.

edge_score = annualized_ROC x confidence_multiplier x liquidity_factor

`maker_price` must be the price we'd actually get filled at as a maker
(the bid we intend to post), not the mid -- see §2.3. `estimated_true_probability`
is whatever the classifier/operator's own model of q is; this module only
enforces the margin-of-error gate on top of it, it doesn't produce q itself.
"""

from __future__ import annotations

from dataclasses import dataclass

from poly03.config import BOOK_A_PRICE_BAND, LIQUIDITY_SIZE_FRACTION_CAP, MIN_MARGIN_PP
from poly03.scoring.roc import annualized_roc


@dataclass
class EdgeScoreInputs:
    market_id: str
    maker_price: float
    days_to_resolution: float
    confidence_multiplier: float
    target_size_usd: float
    visible_depth_usd: float
    estimated_true_probability: float


@dataclass
class EdgeScoreResult:
    market_id: str
    annualized_roc: float
    liquidity_factor: float
    edge_score: float
    margin_pp: float
    passes_min_margin: bool
    passes_price_band: bool

    @property
    def tradeable(self) -> bool:
        return self.passes_min_margin and self.passes_price_band and self.edge_score > 0


def liquidity_factor(target_size_usd: float, visible_depth_usd: float) -> float:
    """1.0 while our target size is within LIQUIDITY_SIZE_FRACTION_CAP of
    visible depth; decays beyond that rather than hard-cutting, since size
    can usually be trimmed to fit instead of skipping the market outright."""
    if visible_depth_usd <= 0:
        return 0.0
    fraction = target_size_usd / visible_depth_usd
    if fraction <= LIQUIDITY_SIZE_FRACTION_CAP:
        return 1.0
    return max(0.0, LIQUIDITY_SIZE_FRACTION_CAP / fraction)


def compute_edge_score(inputs: EdgeScoreInputs) -> EdgeScoreResult:
    roc = annualized_roc(inputs.maker_price, inputs.days_to_resolution)
    liq = liquidity_factor(inputs.target_size_usd, inputs.visible_depth_usd)
    score = roc * inputs.confidence_multiplier * liq

    margin_pp = inputs.estimated_true_probability - inputs.maker_price
    lo, hi = BOOK_A_PRICE_BAND

    return EdgeScoreResult(
        market_id=inputs.market_id,
        annualized_roc=roc,
        liquidity_factor=liq,
        edge_score=score,
        margin_pp=margin_pp,
        passes_min_margin=margin_pp >= MIN_MARGIN_PP,
        passes_price_band=lo <= inputs.maker_price <= hi,
    )
