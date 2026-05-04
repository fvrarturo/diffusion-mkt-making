"""Bootstrap rank-correlation test (proposal §5.2).

Central hypothesis: rankings produced by SYNTHETIC stress regimes predict
rankings on HELD-OUT REAL days better than HISTORICAL replay does.

  H1:  ρ(π_diff, π_truth) > ρ(π_hist, π_truth)

We test by bootstrap-resampling the holdout days (with replacement),
recomputing π_truth on each resample, and computing both ρ values. The
one-sided p-value is the fraction of bootstrap resamples in which the
inequality FAILS (i.e. evidence against H1).

Inputs:
  - per-day metrics for HOLDOUT days (so we can re-rank π_truth on resamples)
  - the fixed π_hist and π_diff rankings (computed once on val/synth)
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .metrics import aggregate
from .ranking import Ranking, _rank_by, spearman_rho


@dataclass(frozen=True)
class BootstrapResult:
    """Results of the central hypothesis test."""
    n_boot: int
    rho_hist_point: float                 # ρ(π_hist, π_truth) on the actual holdout
    rho_diff_point: float
    rho_hist_ci: tuple[float, float]      # 95% bootstrap CI
    rho_diff_ci: tuple[float, float]
    rho_hist_samples: np.ndarray          # all bootstrap rho values
    rho_diff_samples: np.ndarray
    p_value_one_sided: float              # P(ρ_diff <= ρ_hist) over bootstrap


def _rank_truth(holdout_rows: list[dict]) -> Ranking:
    """π_truth: rank by Sharpe on (a sample of) holdout days, same metric as π_hist."""
    by_agent: dict[str, list[dict]] = defaultdict(list)
    for row in holdout_rows:
        by_agent[row["agent"]].append(row)
    scores = {a: aggregate(rows).sharpe for a, rows in by_agent.items()}
    return Ranking(_rank_by(scores, True), scores, "sharpe", True)


def bootstrap_central_test(
    holdout_per_day: list[dict],
    pi_hist: Ranking,
    pi_diff: Ranking,
    *,
    n_boot: int = 1000,
    seed: int = 42,
) -> BootstrapResult:
    """Run the central hypothesis bootstrap.

    `holdout_per_day` is one row per (agent, holdout_day). The bootstrap
    resamples DAYS (not rows individually) so each agent's PnL distribution
    on a resampled day is preserved together.
    """
    # Group rows by tape_path so we can resample at the day level.
    by_day: dict[str, list[dict]] = defaultdict(list)
    for row in holdout_per_day:
        by_day[row["tape_path"]].append(row)
    days = sorted(by_day.keys())
    n_days = len(days)
    if n_days < 2:
        raise ValueError(f"need ≥2 holdout days for bootstrap, got {n_days}")

    # Point estimate on the actual (un-resampled) holdout.
    pi_truth = _rank_truth(holdout_per_day)
    rho_hist_point = spearman_rho(pi_hist, pi_truth)
    rho_diff_point = spearman_rho(pi_diff, pi_truth)

    rng = np.random.default_rng(seed)
    rho_hist_samples = np.empty(n_boot, dtype=np.float64)
    rho_diff_samples = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n_days, size=n_days)
        sampled_rows: list[dict] = []
        for i in idx:
            sampled_rows.extend(by_day[days[i]])
        try:
            pi_truth_b = _rank_truth(sampled_rows)
            rho_hist_samples[b] = spearman_rho(pi_hist, pi_truth_b)
            rho_diff_samples[b] = spearman_rho(pi_diff, pi_truth_b)
        except (ValueError, ZeroDivisionError):
            # E.g. all agents tied → undefined rho. Fall back to 0.
            rho_hist_samples[b] = 0.0
            rho_diff_samples[b] = 0.0

    rho_hist_ci = (
        float(np.quantile(rho_hist_samples, 0.025)),
        float(np.quantile(rho_hist_samples, 0.975)),
    )
    rho_diff_ci = (
        float(np.quantile(rho_diff_samples, 0.025)),
        float(np.quantile(rho_diff_samples, 0.975)),
    )

    # One-sided p-value: P(ρ_diff <= ρ_hist) over bootstrap resamples.
    # Small p-value → evidence FOR H1 (ρ_diff > ρ_hist).
    p_value = float(np.mean(rho_diff_samples <= rho_hist_samples))

    return BootstrapResult(
        n_boot=n_boot,
        rho_hist_point=rho_hist_point,
        rho_diff_point=rho_diff_point,
        rho_hist_ci=rho_hist_ci,
        rho_diff_ci=rho_diff_ci,
        rho_hist_samples=rho_hist_samples,
        rho_diff_samples=rho_diff_samples,
        p_value_one_sided=p_value,
    )
