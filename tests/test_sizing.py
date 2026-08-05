from __future__ import annotations

import pytest

from poly03.classifier.taxonomy import Tier
from poly03.sizing.position_sizing import (
    SizingInputs,
    check_portfolio_caps,
    compute_stake,
    kelly_fraction,
)


def test_kelly_fraction_matches_doc_worked_example():
    # strategy_v1.md §4.1: p=0.92, q=0.99 -> Kelly says bet 87.5% of bankroll
    assert kelly_fraction(q=0.99, p=0.92) == pytest.approx(0.875, abs=1e-6)


def test_kelly_fraction_zero_when_no_edge():
    assert kelly_fraction(q=0.90, p=0.92) == 0.0


def test_kelly_fraction_clamped_to_one():
    assert kelly_fraction(q=0.999999, p=0.01) <= 1.0


def test_compute_stake_kelly_cap_binds_when_kelly_is_extreme():
    result = compute_stake(
        SizingInputs(
            bankroll=100_000,
            tier=Tier.TIER_1,
            maker_price=0.92,
            estimated_true_probability=0.99,  # full Kelly = 87.5%, 1/10 Kelly = 8.75%
            visible_book_depth_usd=1_000_000,  # depth cap not binding
        )
    )
    # base_fraction (0.5% * 2.0x tier1 = 1%) and max_position (2%) are both
    # far below the 1/10-Kelly value here, so one of those should bind, not Kelly
    assert result.binding_constraint in ("base_fraction", "max_position")
    assert result.stake_usd < result.components["kelly_capped"]


def test_compute_stake_depth_cap_binds_on_thin_book():
    result = compute_stake(
        SizingInputs(
            bankroll=1_000_000,
            tier=Tier.TIER_1,
            maker_price=0.92,
            estimated_true_probability=0.99,
            visible_book_depth_usd=500,  # 10% of this is tiny relative to bankroll-scaled caps
        )
    )
    assert result.binding_constraint == "depth_cap"
    assert result.stake_usd == pytest.approx(50.0)


def test_tier4_multiplier_zeroes_out_base_fraction():
    result = compute_stake(
        SizingInputs(
            bankroll=100_000,
            tier=Tier.TIER_4,
            maker_price=0.92,
            estimated_true_probability=0.99,
            visible_book_depth_usd=1_000_000,
        )
    )
    assert result.components["base_fraction"] == 0.0


def test_portfolio_caps_reject_over_deployed_book_a():
    check = check_portfolio_caps(
        book="A",
        bankroll=100_000,
        cash_available=30_000,
        book_a_deployed_usd=59_000,
        book_b_deployed_usd=0,
        proposed_stake_usd=2_000,
    )
    assert not check.ok
    assert any("Book A" in r for r in check.reasons)


def test_portfolio_caps_reject_cash_reserve_breach():
    check = check_portfolio_caps(
        book="A",
        bankroll=100_000,
        cash_available=21_000,
        book_a_deployed_usd=0,
        book_b_deployed_usd=0,
        proposed_stake_usd=2_000,
    )
    assert not check.ok
    assert any("cash reserve" in r for r in check.reasons)


def test_portfolio_caps_ok_within_limits():
    check = check_portfolio_caps(
        book="A",
        bankroll=100_000,
        cash_available=50_000,
        book_a_deployed_usd=10_000,
        book_b_deployed_usd=0,
        proposed_stake_usd=1_000,
    )
    assert check.ok
    assert check.reasons == []
