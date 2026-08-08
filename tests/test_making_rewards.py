from __future__ import annotations

import pytest

from poly03.data.models import OrderBook
from poly03.making.rewards import (
    RewardConfig,
    book_qscore,
    combine_sides,
    estimate_share,
    order_score,
)


def _cfg(min_size=20.0, max_spread=4.5, daily=35.0):
    return RewardConfig(min_size=min_size, max_spread_cents=max_spread, daily_rate_usd=daily)


# --- config parsing ---------------------------------------------------------


def test_from_clob_market_parses_the_rewards_block():
    raw = {
        "rewards": {
            "rates": [{"rewards_daily_rate": 35}],
            "min_size": 20,
            "max_spread": 4.5,
        }
    }
    cfg = RewardConfig.from_clob_market(raw)
    assert cfg == RewardConfig(min_size=20.0, max_spread_cents=4.5, daily_rate_usd=35.0)


def test_from_clob_market_sums_multiple_rate_entries():
    raw = {"rewards": {"rates": [{"rewards_daily_rate": 10}, {"rewards_daily_rate": 25}], "min_size": 20, "max_spread": 4.5}}
    assert RewardConfig.from_clob_market(raw).daily_rate_usd == 35.0


@pytest.mark.parametrize(
    "rewards",
    [
        {"rates": [], "min_size": 0, "max_spread": 0},          # listed but unfunded
        {"rates": [{"rewards_daily_rate": 0}], "min_size": 20, "max_spread": 4.5},
        {"rates": [{"rewards_daily_rate": 35}], "min_size": 20, "max_spread": 0},
    ],
)
def test_from_clob_market_rejects_unfunded_configs(rewards):
    assert RewardConfig.from_clob_market({"rewards": rewards}) is None


# --- order scoring ----------------------------------------------------------


def test_order_at_the_midpoint_scores_full_size():
    assert order_score(0.50, 100.0, 0.50, _cfg()) == pytest.approx(100.0)


def test_score_decays_toward_the_spread_limit():
    cfg = _cfg(max_spread=4.0)
    near = order_score(0.49, 100.0, 0.50, cfg)   # 1c out
    far = order_score(0.47, 100.0, 0.50, cfg)    # 3c out
    assert near > far > 0


def test_order_outside_max_spread_scores_zero():
    # 5c from mid with a 4.5c window
    assert order_score(0.45, 100.0, 0.50, _cfg()) == 0.0


def test_order_below_min_size_scores_zero():
    """The min_size cliff is why quoting.py posts full size or nothing."""
    assert order_score(0.50, 19.0, 0.50, _cfg(min_size=20.0)) == 0.0


# --- two-sided combination --------------------------------------------------


def test_one_sided_quote_scores_zero_in_the_middle_of_the_book():
    assert combine_sides(100.0, 0.0, midpoint=0.50) == 0.0


def test_lopsided_quote_is_capped_at_twice_the_smaller_side():
    assert combine_sides(300.0, 50.0, midpoint=0.50) == pytest.approx(100.0)


def test_balanced_quote_scores_the_sum():
    assert combine_sides(100.0, 80.0, midpoint=0.50) == pytest.approx(180.0)


def test_one_sided_quote_qualifies_at_the_tails():
    assert combine_sides(100.0, 0.0, midpoint=0.95) == pytest.approx(100.0)
    assert combine_sides(0.0, 100.0, midpoint=0.05) == pytest.approx(100.0)


# --- book scoring and share -------------------------------------------------


def test_book_qscore_counts_only_depth_inside_the_window():
    book = OrderBook(
        asset_id="t",
        bids=[{"price": 0.49, "size": 100}, {"price": 0.30, "size": 5_000}],
        asks=[{"price": 0.51, "size": 100}, {"price": 0.70, "size": 5_000}],
    )
    # the 0.30/0.70 levels are 20c out and must not contribute
    assert book_qscore(book, 0.50, _cfg()) == pytest.approx(2 * order_score(0.49, 100.0, 0.50, _cfg()))


def test_book_qscore_does_not_apply_the_per_maker_two_sided_penalty():
    """A lopsided *book* is many makers, not one lopsided maker. Applying
    combine_sides here collapsed the denominator and inflated our share."""
    book = OrderBook(
        asset_id="t",
        bids=[{"price": 0.49, "size": 1_000}],
        asks=[{"price": 0.51, "size": 50}],
    )
    q = book_qscore(book, 0.50, _cfg())
    # combine_sides would cap this at 2x the smaller side
    assert q > 2 * order_score(0.51, 50.0, 0.50, _cfg())


def test_book_qscore_counts_levels_below_min_size():
    """min_size applies per order; an aggregated level below it may still be
    several qualifying makers. Zeroing them deleted most of our competition."""
    book = OrderBook(
        asset_id="t",
        bids=[{"price": 0.49, "size": 10}],
        asks=[{"price": 0.51, "size": 10}],
    )
    assert book_qscore(book, 0.50, _cfg(min_size=200.0)) > 0


def test_estimate_share_is_pro_rata():
    est = estimate_share(our_q=25.0, competing_q=75.0, cfg=_cfg(daily=100.0))
    assert est.share_fraction == pytest.approx(0.25)
    assert est.usd_per_day == pytest.approx(25.0)
    assert est.identified
    assert not est.capped


def test_estimate_share_of_nothing_is_nothing():
    est = estimate_share(0.0, 100.0, _cfg())
    assert est.share_fraction == 0.0
    assert est.usd_per_day == 0.0


def test_negligible_competition_is_unidentified():
    """Non-zero but sub-minimum competing score is still no evidence: a pool
    that easy to capture would not stay uncontested."""
    est = estimate_share(our_q=4.4, competing_q=0.5, cfg=_cfg(min_size=20.0, daily=126.0))
    assert not est.identified


def test_one_qualifying_competitor_is_enough_to_identify():
    est = estimate_share(our_q=50.0, competing_q=20.0, cfg=_cfg(min_size=20.0))
    assert est.identified


def test_uncontested_pool_is_flagged_unidentified_and_capped():
    """No competing depth inside the window means the raw share is 100% for
    any size at all -- absence of evidence, not a finding. It must be flagged
    and capped, or it dominates the headline number."""
    est = estimate_share(our_q=0.5, competing_q=0.0, cfg=_cfg(daily=126.0))
    assert est.raw_share_fraction == pytest.approx(1.0)
    assert not est.identified
    assert est.capped
    assert est.share_fraction == pytest.approx(0.50)
    assert est.usd_per_day == pytest.approx(63.0)


def test_dominant_but_contested_share_is_still_capped():
    est = estimate_share(our_q=900.0, competing_q=100.0, cfg=_cfg(daily=100.0))
    assert est.identified
    assert est.capped
    assert est.share_fraction == pytest.approx(0.50)
