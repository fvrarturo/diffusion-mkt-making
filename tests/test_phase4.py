"""Phase 4 smoke tests — metric aggregation, agent rankings, Spearman ρ.

Plus an end-to-end mini-pipeline: calibrate → replay → rank, on synthetic
canonical tapes built by tests/test_phase3.py's factory.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from diffmm.eval.metrics import aggregate
from diffmm.eval.ranking import (
    Ranking, rank_by_sharpe, rank_by_worst_quartile, spearman_rho,
)


# ─── aggregation ───────────────────────────────────────────────────────

def test_aggregate_basic_sharpe_and_cvar():
    """Aggregate matches hand-computed values for a tiny sample."""
    rows = [
        {"agent": "A1", "terminal_pnl": 100.0, "inventory_var": 1.0, "fill_rate": 0.1},
        {"agent": "A1", "terminal_pnl": 200.0, "inventory_var": 1.5, "fill_rate": 0.2},
        {"agent": "A1", "terminal_pnl": -50.0, "inventory_var": 0.5, "fill_rate": 0.05},
        {"agent": "A1", "terminal_pnl": 75.0,  "inventory_var": 0.7, "fill_rate": 0.15},
    ]
    agg = aggregate(rows)
    assert agg.agent == "A1"
    assert agg.n_days == 4
    assert agg.mean_pnl == pytest.approx(81.25)
    pnls = np.array([100.0, 200.0, -50.0, 75.0])
    expected_sharpe = math.sqrt(252) * pnls.mean() / pnls.std(ddof=1)
    assert agg.sharpe == pytest.approx(expected_sharpe, rel=1e-9)


def test_aggregate_rejects_mixed_agents():
    rows = [
        {"agent": "A1", "terminal_pnl": 1.0, "inventory_var": 0, "fill_rate": 0},
        {"agent": "A2", "terminal_pnl": 1.0, "inventory_var": 0, "fill_rate": 0},
    ]
    with pytest.raises(AssertionError):
        aggregate(rows)


# ─── ranking ───────────────────────────────────────────────────────────

def test_rank_by_sharpe_orders_descending():
    rows = []
    # A1: high mean low std → highest Sharpe
    rows += [{"agent": "A1", "terminal_pnl": p, "inventory_var": 0, "fill_rate": 0}
             for p in (100, 110, 105, 95, 100)]
    # A2: zero mean → ~zero Sharpe
    rows += [{"agent": "A2", "terminal_pnl": p, "inventory_var": 0, "fill_rate": 0}
             for p in (10, -10, 5, -5, 0)]
    # A3: negative mean
    rows += [{"agent": "A3", "terminal_pnl": p, "inventory_var": 0, "fill_rate": 0}
             for p in (-50, -60, -55, -45, -50)]

    rank = rank_by_sharpe(rows)
    assert rank.agents == ("A1", "A2", "A3")
    assert rank.metric == "sharpe"


def test_rank_by_worst_quartile_uses_25th_percentile():
    rows = []
    # A1: a fat bad-day quartile (4 of 10 days negative). q=0.25 lands in it.
    rows += [{"agent": "A1", "terminal_pnl": p, "inventory_var": 0, "fill_rate": 0}
             for p in (-1000, -500, -300, -200, 50, 50, 50, 50, 50, 50)]
    # A2: stable, no negatives. q=0.25 = 50.
    rows += [{"agent": "A2", "terminal_pnl": p, "inventory_var": 0, "fill_rate": 0}
             for p in (50, 50, 50, 50, 50, 50, 50, 50, 50, 50)]

    rank = rank_by_worst_quartile(rows)
    # A2's worst-quartile (50) beats A1's (negative), even though means could
    # be similar. This is exactly why π_diff uses worst-quartile, not mean.
    assert rank.agents == ("A2", "A1")


# ─── Spearman ρ ────────────────────────────────────────────────────────

def test_spearman_identical_rankings_is_one():
    r1 = Ranking(("A", "B", "C"), {"A": 3.0, "B": 2.0, "C": 1.0}, "x", True)
    r2 = Ranking(("A", "B", "C"), {"A": 30, "B": 20, "C": 10}, "y", True)
    assert spearman_rho(r1, r2) == pytest.approx(1.0)


def test_spearman_reversed_rankings_is_minus_one():
    r1 = Ranking(("A", "B", "C"), {}, "x")
    r2 = Ranking(("C", "B", "A"), {}, "y")
    assert spearman_rho(r1, r2) == pytest.approx(-1.0)


def test_spearman_partial_agreement():
    # Agents in order: real (A B C D), test (A C B D) → one swap in middle
    r_real = Ranking(("A", "B", "C", "D"), {}, "x")
    r_test = Ranking(("A", "C", "B", "D"), {}, "y")
    # Spearman: positions a:1 vs 1, b:2 vs 3, c:3 vs 2, d:4 vs 4
    # d^2 sum = 0 + 1 + 1 + 0 = 2
    # rho = 1 - 6*2/(4*15) = 1 - 12/60 = 0.8
    assert spearman_rho(r_real, r_test) == pytest.approx(0.8)


def test_spearman_rejects_disjoint_agent_sets():
    r1 = Ranking(("A", "B"), {}, "x")
    r2 = Ranking(("A", "C"), {}, "y")
    with pytest.raises(ValueError):
        spearman_rho(r1, r2)


def test_bootstrap_central_test_runs_and_returns_p_value():
    """Bootstrap returns sane shapes + a p-value in [0, 1]."""
    import numpy as np
    from diffmm.eval.bootstrap import bootstrap_central_test

    # Synthesize 20 holdout days × 3 agents. A1 wins on holdout consistently.
    rows = []
    rng = np.random.default_rng(0)
    for day in range(20):
        rows.append({"agent": "A1", "tape_path": f"day_{day}.parquet",
                     "terminal_pnl": 100.0 + rng.normal(0, 10),
                     "inventory_var": 1.0, "fill_rate": 0.1})
        rows.append({"agent": "A2", "tape_path": f"day_{day}.parquet",
                     "terminal_pnl": 50.0 + rng.normal(0, 10),
                     "inventory_var": 1.0, "fill_rate": 0.1})
        rows.append({"agent": "A3", "tape_path": f"day_{day}.parquet",
                     "terminal_pnl": -20.0 + rng.normal(0, 10),
                     "inventory_var": 1.0, "fill_rate": 0.1})

    pi_hist = Ranking(("A1", "A2", "A3"), {}, "sharpe")
    pi_diff = Ranking(("A1", "A2", "A3"), {}, "sharpe")  # identical

    res = bootstrap_central_test(rows, pi_hist, pi_diff, n_boot=200, seed=0)
    assert res.n_boot == 200
    assert -1.0 <= res.rho_hist_point <= 1.0
    assert -1.0 <= res.rho_diff_point <= 1.0
    assert 0.0 <= res.p_value_one_sided <= 1.0
    assert res.rho_hist_samples.shape == (200,)
    # Identical π_hist and π_diff → identical bootstrap samples → p ≈ 1.0
    assert res.p_value_one_sided > 0.9


def test_bootstrap_detects_when_diff_clearly_better():
    """If π_diff matches π_truth and π_hist disagrees, p-value should be small."""
    import numpy as np
    from diffmm.eval.bootstrap import bootstrap_central_test

    rng = np.random.default_rng(1)
    rows = []
    # Holdout truth: A3 best, A1 worst (consistent across days).
    for day in range(20):
        for agent, base in [("A1", -50), ("A2", 0), ("A3", 100)]:
            rows.append({"agent": agent, "tape_path": f"day_{day}.parquet",
                         "terminal_pnl": base + rng.normal(0, 5),
                         "inventory_var": 1.0, "fill_rate": 0.1})

    pi_hist = Ranking(("A1", "A2", "A3"), {}, "sharpe")    # backwards from truth
    pi_diff = Ranking(("A3", "A2", "A1"), {}, "sharpe")    # matches truth

    res = bootstrap_central_test(rows, pi_hist, pi_diff, n_boot=500, seed=1)
    assert res.rho_diff_point > res.rho_hist_point
    assert res.p_value_one_sided < 0.05, (
        f"expected p<0.05 but got {res.p_value_one_sided}; "
        f"ρ_diff={res.rho_diff_point}, ρ_hist={res.rho_hist_point}"
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
