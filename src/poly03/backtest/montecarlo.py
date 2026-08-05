"""§4.2: "Model the equity path with a Monte Carlo before going live,
including a stress case where losses cluster (they will -- see §4.3). If
the drawdown in the 5th-percentile path is unacceptable, the sizing is
wrong, not the model."

cluster_stress_prob/cluster_stress_loss_fraction let you inject exactly
that correlated-loss stress case: with probability cluster_stress_prob a
cluster-wide shock hits, and cluster_stress_loss_fraction of that
cluster's *otherwise-winning* positions flip to losses together --
approximating "one regime event marks the whole book at once" rather than
treating every position as independent.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from poly03.scoring.roc import gross_return


@dataclass
class PositionSpec:
    price: float
    stake: float
    true_prob: float
    cluster_id: str = "default"


@dataclass
class MonteCarloResult:
    n_sims: int
    final_pnl: np.ndarray
    max_drawdown: np.ndarray
    percentiles: dict[int, float] = field(default_factory=dict)
    drawdown_percentiles: dict[int, float] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [f"Monte Carlo over {self.n_sims} sims:"]
        lines.append("  final P&L percentiles:")
        for p, v in sorted(self.percentiles.items()):
            lines.append(f"    p{p:>2}: {v:+,.2f}")
        lines.append("  max drawdown percentiles (fraction of peak equity):")
        for p, v in sorted(self.drawdown_percentiles.items()):
            lines.append(f"    p{p:>2}: {v:.1%}")
        return "\n".join(lines)


def uniform_book(
    n_positions: int, *, price: float, true_prob: float, stake: float, cluster_size: int | None = None
) -> list[PositionSpec]:
    """Build the doc's canonical example: N positions at a common entry
    price and true resolution rate. cluster_size, if set, groups positions
    into clusters of that size (cluster_0, cluster_1, ...) so
    cluster_stress_prob has something to act on."""
    positions = []
    for i in range(n_positions):
        cluster_id = f"cluster_{i // cluster_size}" if cluster_size else "default"
        positions.append(PositionSpec(price=price, stake=stake, true_prob=true_prob, cluster_id=cluster_id))
    return positions


def simulate_equity_paths(
    positions: list[PositionSpec],
    *,
    bankroll: float,
    n_sims: int = 10_000,
    cluster_stress_prob: float = 0.0,
    cluster_stress_loss_fraction: float = 1.0,
    rng_seed: int | None = None,
) -> MonteCarloResult:
    rng = np.random.default_rng(rng_seed)
    n = len(positions)
    if n == 0:
        raise ValueError("positions must be non-empty")

    prices = np.array([p.price for p in positions])
    stakes = np.array([p.stake for p in positions])
    true_probs = np.array([p.true_prob for p in positions])
    clusters = [p.cluster_id for p in positions]
    unique_clusters = sorted(set(clusters))
    cluster_index = {c: i for i, c in enumerate(unique_clusters)}
    cluster_idx_arr = np.array([cluster_index[c] for c in clusters])
    n_clusters = len(unique_clusters)

    gross_returns = np.array([gross_return(p) for p in prices])

    wins = rng.random((n_sims, n)) < true_probs[None, :]

    if cluster_stress_prob > 0:
        cluster_shock = rng.random((n_sims, n_clusters)) < cluster_stress_prob
        shock_per_position = cluster_shock[:, cluster_idx_arr]
        forced_loss_draw = rng.random((n_sims, n)) < cluster_stress_loss_fraction
        forced_loss = shock_per_position & forced_loss_draw
        wins = wins & ~forced_loss

    pnl_matrix = np.where(wins, stakes[None, :] * gross_returns[None, :], -stakes[None, :])

    order_keys = rng.random((n_sims, n))
    order = np.argsort(order_keys, axis=1)
    pnl_ordered = np.take_along_axis(pnl_matrix, order, axis=1)
    cum_pnl = np.cumsum(pnl_ordered, axis=1)
    equity = bankroll + cum_pnl
    running_max = np.maximum.accumulate(equity, axis=1)
    drawdown_frac = np.divide(
        running_max - equity, running_max, out=np.zeros_like(equity), where=running_max > 0
    )
    max_drawdown = drawdown_frac.max(axis=1)
    final_pnl = cum_pnl[:, -1]

    pcts = (5, 25, 50, 75, 95)
    return MonteCarloResult(
        n_sims=n_sims,
        final_pnl=final_pnl,
        max_drawdown=max_drawdown,
        percentiles={p: float(np.percentile(final_pnl, p)) for p in pcts},
        drawdown_percentiles={p: float(np.percentile(max_drawdown, p)) for p in pcts},
    )
