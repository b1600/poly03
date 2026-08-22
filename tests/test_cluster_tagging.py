from __future__ import annotations

from poly03.cluster.tagging import ClusterExposureTracker, tag_market


def test_entity_extraction_prefers_group_item_title(market_factory):
    m = market_factory("Will this candidate win?", group_item_title="J.D. Vance")
    tags = tag_market(m)
    assert tags.entity == "J.D. Vance"


def test_entity_extraction_falls_back_to_proper_noun(market_factory):
    m = market_factory("Will Xi Jinping leave office?")
    tags = tag_market(m)
    assert "Xi Jinping" in tags.entity


def test_geography_extracted_from_tags(market_factory):
    m = market_factory("Will something happen?", tags=["Elections", "United States", "Politics"])
    tags = tag_market(m)
    assert tags.geography == "United States"


def test_resolution_source_normalized_to_domain(market_factory):
    m = market_factory("Will X happen?", resolution_source="https://hltv.org/matches/123")
    tags = tag_market(m)
    assert tags.resolution_source == "hltv.org"


def test_resolution_source_unknown_when_absent(market_factory):
    m = market_factory("Will X happen?", "no url here", resolution_source="")
    tags = tag_market(m)
    assert tags.resolution_source == "unknown"


def test_entity_cap_breach_and_registration(market_factory):
    tracker = ClusterExposureTracker(bankroll=100_000)
    m = market_factory("Will Trump do X?", group_item_title="Trump")
    tags = tag_market(m)

    # 15% cap = 15,000; a single 20,000 stake should breach it
    assert tracker.would_breach(tags, 20_000) != []

    # two 8,000 stakes (16,000 total) should breach on the second call
    assert tracker.would_breach(tags, 8_000) == []
    tracker.register(tags, 8_000)
    assert tracker.would_breach(tags, 8_000) != []


def test_release_frees_up_capacity(market_factory):
    tracker = ClusterExposureTracker(bankroll=100_000)
    m = market_factory("Will Trump do X?", group_item_title="Trump")
    tags = tag_market(m)

    tracker.register(tags, 14_000)
    assert tracker.would_breach(tags, 2_000) != []
    tracker.release(tags, 14_000)
    assert tracker.would_breach(tags, 2_000) == []


def test_cap_fraction_overrides_apply_instead_of_module_defaults(market_factory):
    """task 20260818_2012 item 1b: execution.py's live engine passes its own,
    much larger MAKING_LIVE_MAX_*_CLUSTER_FRACTION knobs -- confirm the
    tracker actually uses the override, not the §4.3 Phase 0 constant."""
    m = market_factory("Will Trump do X?", group_item_title="Trump")
    tags = tag_market(m)

    default_tracker = ClusterExposureTracker(bankroll=100)
    # 15% of 100 = 15 -- a $20 stake breaches the default entity cap.
    assert default_tracker.would_breach(tags, 20) != []

    live_tracker = ClusterExposureTracker(bankroll=100, entity_cap_fraction=0.35)
    # 35% of 100 = 35 -- the same $20 stake fits under the override.
    assert live_tracker.would_breach(tags, 20) == []
