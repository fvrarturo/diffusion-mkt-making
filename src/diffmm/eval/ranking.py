"""Agent rankings — π_hist, π_diff, π_truth (proposal §5).

  π_hist  : ranking of agents by Sharpe on historical (real) val replay
  π_diff  : ranking by worst-quartile PnL on synthetic stress regimes
  π_truth : ranking by Sharpe on the held-out real days (Phase 5)

Returns ordered tuples of agent names, best (rank 1) first. The central
hypothesis test (Phase 5) compares Spearman rho(π_diff, π_truth) versus
Spearman rho(π_hist, π_truth).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from .metrics import AggregateMetrics, aggregate


@dataclass(frozen=True)
class Ranking:
    """Ordered tuple of agent names plus the per-agent score that ordered them."""
    agents: tuple[str, ...]                  # best (rank 1) first
    scores: dict[str, float]
    metric: str                              # "sharpe" | "worst_quartile_pnl" | etc.
    higher_is_better: bool = True


def _rank_by(scores: dict[str, float], higher_is_better: bool) -> tuple[str, ...]:
    return tuple(sorted(scores, key=lambda a: scores[a], reverse=higher_is_better))


def rank_by_sharpe(per_day_metrics: list[dict]) -> Ranking:
    """π_hist style ranking: aggregate over all days, rank by Sharpe."""
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in per_day_metrics:
        by_agent[row["agent"]].append(row)
    scores = {a: aggregate(rows).sharpe for a, rows in by_agent.items()}
    return Ranking(_rank_by(scores, True), scores, "sharpe", True)


def rank_by_worst_quartile(per_day_metrics: list[dict]) -> Ranking:
    """π_diff style ranking: rank by 25th-percentile PnL (robust-to-tail)."""
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in per_day_metrics:
        by_agent[row["agent"]].append(row)
    scores = {a: aggregate(rows).worst_quartile_pnl for a, rows in by_agent.items()}
    return Ranking(_rank_by(scores, True), scores, "worst_quartile_pnl", True)


def rank_by_mean_pnl(per_day_metrics: list[dict]) -> Ranking:
    """Rank by mean terminal PnL across days. Useful when worst-quartile
    degenerates to zero (e.g. short synthetic windows where most days yield
    near-zero PnL)."""
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in per_day_metrics:
        by_agent[row["agent"]].append(row)
    scores = {a: aggregate(rows).mean_pnl for a, rows in by_agent.items()}
    return Ranking(_rank_by(scores, True), scores, "mean_pnl", True)


def spearman_rho(rank_a: Ranking, rank_b: Ranking) -> float:
    """Spearman rank correlation between two rankings of the SAME agent set.

    Uses position-in-ranking as the rank value (1 = best). Identical
    rankings → +1, fully reversed → -1.
    """
    agents_a = set(rank_a.agents)
    agents_b = set(rank_b.agents)
    if agents_a != agents_b:
        raise ValueError(f"ranking agent sets differ: {agents_a} vs {agents_b}")
    n = len(rank_a.agents)
    if n < 2:
        return 0.0
    pos_a = {agent: i + 1 for i, agent in enumerate(rank_a.agents)}
    pos_b = {agent: i + 1 for i, agent in enumerate(rank_b.agents)}
    d_sq_sum = sum((pos_a[a] - pos_b[a]) ** 2 for a in agents_a)
    return 1.0 - (6.0 * d_sq_sum) / (n * (n * n - 1))
