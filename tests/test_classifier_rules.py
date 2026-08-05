from __future__ import annotations

from poly03.classifier.rules import classify_market_rules
from poly03.classifier.taxonomy import Tier


def test_forecast_shaped_sports_is_tier4(market_factory):
    m = market_factory("Team A vs Team B: who wins?", "This resolves based on the match result.")
    c = classify_market_rules(m)
    assert c.tier == Tier.TIER_4
    assert not c.conservative_fallback  # positively matched a forecast pattern, not a default


def test_electoral_nomination_is_tier4(market_factory):
    m = market_factory("Will Jane Doe win the Democratic nomination?", "Resolves Yes if Jane Doe wins.")
    c = classify_market_rules(m)
    assert c.tier == Tier.TIER_4


def test_price_level_is_tier4(market_factory):
    m = market_factory("Will Bitcoin reach $100,000?", "Resolves Yes if BTC reaches $100k.")
    c = classify_market_rules(m)
    assert c.tier == Tier.TIER_4


def test_unclassifiable_falls_back_to_tier4(market_factory):
    m = market_factory("Will Trump wear a hat in June?", "Resolves Yes if he wears a hat.")
    c = classify_market_rules(m)
    assert c.tier == Tier.TIER_4
    assert c.conservative_fallback  # this is the "unclassifiable -> no trade" path specifically


def test_never_happens_style_is_tier3(market_factory):
    m = market_factory(
        "Jesus Christ returns before 2027?",
        "This market resolves Yes if Jesus Christ returns before Jan 1 2027.",
        days_to_resolution=500,
    )
    c = classify_market_rules(m)
    assert c.tier == Tier.TIER_3


def test_regime_change_with_hard_clock_is_tier2(market_factory):
    m = market_factory(
        "Iranian regime falls by end of month?",
        "Resolves Yes if the Iranian regime falls before the end of the month.",
        days_to_resolution=25,
    )
    c = classify_market_rules(m)
    assert c.tier == Tier.TIER_2


def test_regime_change_without_hard_clock_downgrades_to_tier3(market_factory):
    m = market_factory(
        "Iranian regime falls before 2027?",
        "Resolves Yes if the Iranian regime falls / is removed from power before 2027.",
        days_to_resolution=500,
    )
    c = classify_market_rules(m)
    assert c.tier == Tier.TIER_3


def test_live_process_underway_downgrades_event_pattern_to_tier4(market_factory):
    m = market_factory(
        "Will the president resign by year end?",
        "The president has announced his intention to resign and a vote is scheduled for next week.",
        days_to_resolution=20,
    )
    c = classify_market_rules(m)
    assert c.tier == Tier.TIER_4


def test_structural_impossibility_phrase_is_tier1(market_factory):
    m = market_factory(
        "Will the amendment pass this year?",
        "This requires a constitutional amendment, which needs a two-thirds vote in both chambers.",
        days_to_resolution=200,
    )
    c = classify_market_rules(m)
    assert c.tier == Tier.TIER_1


def test_named_process_lead_time_forces_tier1(market_factory):
    m = market_factory(
        "Will the constitutional amendment be ratified in time?",
        "This market tracks whether the constitutional amendment process completes in the window.",
        days_to_resolution=10,  # far shorter than the 365-day minimum for this named process
    )
    c = classify_market_rules(m)
    assert c.tier == Tier.TIER_1
    assert "structurally impossible" in c.evidence[0]


def test_forecast_veto_beats_tier1_pattern(market_factory):
    # a question that superficially contains a tier-1 phrase but is really
    # a forecast (sports) should still be excluded -- the tier-4 veto runs first.
    m = market_factory(
        "Team A vs Team B: has the match not yet started?",
        "This requires a two-thirds vote of the referees, which has not been scheduled.",
    )
    c = classify_market_rules(m)
    assert c.tier == Tier.TIER_4
