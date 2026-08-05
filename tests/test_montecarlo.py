from __future__ import annotations

from poly03.backtest.montecarlo import simulate_equity_paths, uniform_book


def test_independent_book_has_low_tail_risk():
    positions = uniform_book(100, price=0.91, true_prob=0.95, stake=500)
    result = simulate_equity_paths(positions, bankroll=100_000, n_sims=5_000, rng_seed=1)
    # at 95% true win rate with no correlation, the 5th percentile outcome
    # should still be solidly non-negative
    assert result.percentiles[5] >= 0


def test_cluster_stress_fattens_the_left_tail():
    positions = uniform_book(100, price=0.91, true_prob=0.95, stake=500, cluster_size=20)
    independent = simulate_equity_paths(positions, bankroll=100_000, n_sims=5_000, rng_seed=1)
    stressed = simulate_equity_paths(
        positions,
        bankroll=100_000,
        n_sims=5_000,
        cluster_stress_prob=0.05,
        cluster_stress_loss_fraction=0.8,
        rng_seed=1,
    )
    assert stressed.percentiles[5] < independent.percentiles[5]
    assert stressed.drawdown_percentiles[95] > independent.drawdown_percentiles[95]


def test_result_summary_is_a_nonempty_string():
    positions = uniform_book(10, price=0.9, true_prob=0.95, stake=100)
    result = simulate_equity_paths(positions, bankroll=10_000, n_sims=1_000, rng_seed=0)
    assert "Monte Carlo" in result.summary()
