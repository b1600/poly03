from __future__ import annotations

import pytest

from poly03.scoring.edge_score import EdgeScoreInputs, compute_edge_score, liquidity_factor
from poly03.scoring.roc import annualized_roc, breakeven_win_rate, gross_return


def test_gross_return_matches_doc_table():
    assert gross_return(0.85) == pytest.approx(0.1765, abs=1e-3)
    assert gross_return(0.90) == pytest.approx(0.1111, abs=1e-3)
    assert gross_return(0.95) == pytest.approx(0.0526, abs=1e-3)
    assert gross_return(0.97) == pytest.approx(0.0309, abs=1e-3)


def test_breakeven_win_rate_is_the_price():
    assert breakeven_win_rate(0.92) == 0.92


def test_annualized_roc_collapses_to_gross_return_at_365_days():
    for p in (0.85, 0.90, 0.95, 0.97):
        assert annualized_roc(p, 365) == pytest.approx(gross_return(p), abs=1e-6)


def test_annualized_roc_rejects_invalid_price():
    with pytest.raises(ValueError):
        annualized_roc(1.5, 30)
    with pytest.raises(ValueError):
        annualized_roc(0.9, 0)


def test_liquidity_factor_full_below_cap():
    assert liquidity_factor(target_size_usd=1_000, visible_depth_usd=100_000) == 1.0


def test_liquidity_factor_penalizes_beyond_cap():
    f = liquidity_factor(target_size_usd=50_000, visible_depth_usd=100_000)  # 50% of depth
    assert 0 < f < 1.0


def test_liquidity_factor_zero_depth():
    assert liquidity_factor(target_size_usd=100, visible_depth_usd=0) == 0.0


def test_edge_score_price_band_and_margin_gate():
    # below the 0.85 floor -> fails price band even with ample margin
    below_band = compute_edge_score(
        EdgeScoreInputs(
            market_id="m1",
            maker_price=0.80,
            days_to_resolution=60,
            confidence_multiplier=1.0,
            target_size_usd=100,
            visible_depth_usd=10_000,
            estimated_true_probability=0.95,
        )
    )
    assert not below_band.passes_price_band
    assert not below_band.tradeable

    # in-band, but margin too thin (< 4pp)
    thin_margin = compute_edge_score(
        EdgeScoreInputs(
            market_id="m2",
            maker_price=0.92,
            days_to_resolution=60,
            confidence_multiplier=1.0,
            target_size_usd=100,
            visible_depth_usd=10_000,
            estimated_true_probability=0.93,
        )
    )
    assert thin_margin.passes_price_band
    assert not thin_margin.passes_min_margin
    assert not thin_margin.tradeable

    # in-band, sufficient margin -> tradeable
    good = compute_edge_score(
        EdgeScoreInputs(
            market_id="m3",
            maker_price=0.92,
            days_to_resolution=60,
            confidence_multiplier=1.0,
            target_size_usd=100,
            visible_depth_usd=10_000,
            estimated_true_probability=0.99,
        )
    )
    assert good.passes_price_band
    assert good.passes_min_margin
    assert good.tradeable
    assert good.edge_score > 0
