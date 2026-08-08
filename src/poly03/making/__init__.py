"""Book M (strategy_v2.md §3): reward-subsidized two-sided making.

The v1 thesis (Book A) needed an estimate of `q`, the true probability, and
never had one -- see strategy_v2.md §1.1. Book M is built so that it doesn't
need one: it earns from liquidity rewards, the maker rebate, and spread
capture, all of which are paid for *placing orders* rather than for being
right. What it does need is to not be badly adversely selected, which is a
bounded and observable problem.

Nothing in this package places a live order. Phase 0 (§4) is measurement only:
it reconstructs what our quotes would have been and what share of the reward
pool they would have earned.
"""
