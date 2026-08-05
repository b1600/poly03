"""§1.1: the arithmetic of Book A. Buy at price p, hold to resolution,
collect $1 if right, $0 if wrong."""

from __future__ import annotations


def gross_return(p: float) -> float:
    """(1 - p) / p -- the simple return of one resolved trade."""
    if not 0 < p < 1:
        raise ValueError(f"price must be in (0, 1), got {p}")
    return (1 - p) / p


def breakeven_win_rate(p: float) -> float:
    """Exactly p -- the win rate at which this trade breaks even."""
    return p


def annualized_roc(p: float, days_to_resolution: float) -> float:
    """(1/p)^(365/T) - 1. Rank every Book A candidate by this, never by
    raw spread to $1 -- see strategy_v1.md §1.1 point 1."""
    if not 0 < p < 1:
        raise ValueError(f"price must be in (0, 1), got {p}")
    if days_to_resolution <= 0:
        raise ValueError(f"days_to_resolution must be > 0, got {days_to_resolution}")
    return (1.0 / p) ** (365.0 / days_to_resolution) - 1.0
