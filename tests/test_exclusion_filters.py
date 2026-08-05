from __future__ import annotations

from poly03.filters.exclusion import (
    ExclusionReason,
    apply_exclusion_filters,
    check_ambiguous_resolution,
    check_early_partial_resolution,
    check_horizon_cap,
    check_resolving_or_dispute,
    check_thin_book,
    check_unreliable_source,
    check_wide_spread,
)


def test_ambiguous_resolution_keyword_detected(market_factory):
    m = market_factory("Will X happen?", "This will resolve based on what is widely reported.")
    hits = check_ambiguous_resolution(m)
    assert hits


def test_clean_resolution_text_has_no_ambiguous_hits(market_factory):
    m = market_factory("Will X happen?", "This resolves per the official government release.", resolution_source="https://example.gov")
    assert check_ambiguous_resolution(m) == []


def test_unreliable_source_true_when_blank_and_no_named_source(market_factory):
    m = market_factory("Will X happen?", "Something will happen if conditions are met.")
    assert check_unreliable_source(m) is True


def test_unreliable_source_false_when_url_present(market_factory):
    m = market_factory("Will X happen?", "Resolution: see https://official-source.example/results")
    assert check_unreliable_source(m) is False


def test_unreliable_source_false_when_resolution_source_field_set(market_factory):
    m = market_factory("Will X happen?", "no url in text", resolution_source="https://hltv.org")
    assert check_unreliable_source(m) is False


def test_early_partial_resolution_detected(market_factory):
    m = market_factory("Will X happen?", "This market may resolve early if the outcome becomes clear.")
    assert check_early_partial_resolution(m) is True


def test_thin_book_fails_when_open_interest_unknown(market_factory):
    m = market_factory("Will X happen?", open_interest=None)
    assert check_thin_book(m) is True


def test_thin_book_fails_below_threshold(market_factory):
    m = market_factory("Will X happen?", open_interest=100.0, volume_24hr=10.0)
    assert check_thin_book(m) is True


def test_thin_book_passes_above_threshold(market_factory):
    m = market_factory("Will X happen?", open_interest=50_000.0, volume_24hr=5_000.0)
    assert check_thin_book(m) is False


def test_wide_spread_fails_when_bid_ask_unknown(market_factory):
    m = market_factory("Will X happen?", best_bid=None, best_ask=None)
    assert check_wide_spread(m) is True


def test_wide_spread_fails_beyond_threshold(market_factory):
    m = market_factory("Will X happen?", best_bid=0.90, best_ask=0.94)
    assert check_wide_spread(m) is True


def test_wide_spread_passes_within_threshold(market_factory):
    m = market_factory("Will X happen?", best_bid=0.92, best_ask=0.93)
    assert check_wide_spread(m) is False


def test_horizon_cap_fails_beyond_default(market_factory):
    m = market_factory("Will X happen?", days_to_resolution=800)
    assert check_horizon_cap(m) is True


def test_horizon_cap_passes_within_default(market_factory):
    m = market_factory("Will X happen?", days_to_resolution=100)
    assert check_horizon_cap(m) is False


def test_resolving_or_dispute_true_when_closed_unresolved(market_factory):
    m = market_factory("Will X happen?", closed=True, uma_resolution_status=None)
    assert check_resolving_or_dispute(m) is True


def test_resolving_or_dispute_false_when_closed_and_resolved(market_factory):
    m = market_factory(
        "Will X happen?", closed=True, uma_resolution_status="resolved", outcome_prices=(1.0, 0.0)
    )
    assert check_resolving_or_dispute(m) is False


def test_apply_exclusion_filters_aggregates_all_reasons(market_factory):
    m = market_factory(
        "Will X happen?",
        "This will resolve based on what is widely reported.",
        open_interest=None,
        best_bid=None,
        best_ask=None,
        days_to_resolution=800,
    )
    result = apply_exclusion_filters(m)
    assert result.excluded
    assert ExclusionReason.AMBIGUOUS_RESOLUTION in result.reasons
    assert ExclusionReason.THIN_BOOK in result.reasons
    assert ExclusionReason.WIDE_SPREAD in result.reasons
    assert ExclusionReason.HORIZON_CAP in result.reasons


def test_apply_exclusion_filters_clean_market_passes(market_factory):
    m = market_factory(
        "Will X happen?",
        "This resolves per the official government release.",
        resolution_source="https://example.gov",
        open_interest=50_000.0,
        volume_24hr=5_000.0,
        best_bid=0.92,
        best_ask=0.93,
        days_to_resolution=100,
        closed=False,
        accepting_orders=True,
    )
    result = apply_exclusion_filters(m)
    assert not result.excluded
