"""Phase 5 §5.2 — central rank-correlation hypothesis test.

Reads the three rankings produced earlier and runs the bootstrap test:
  H1:  ρ(π_diff, π_truth) > ρ(π_hist, π_truth)

Inputs (must already exist):
  results/backtests/real/{ticker}/ranking_pi_hist.json
  results/backtests/synthetic/{ticker}/ranking_pi_diff.json
  results/backtests/holdout/{ticker}/metrics.csv  ← from scripts/40_open_holdout.py

Outputs:
  results/hypothesis/{ticker}/bootstrap.json    (point estimates, CIs, p-value)
  results/hypothesis/{ticker}/figure7.pdf       (proposal Figure 7)
  results/hypothesis/{ticker}/figure7.png
"""
from __future__ import annotations

import json
from pathlib import Path

import hydra
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from omegaconf import DictConfig

from diffmm.eval.bootstrap import bootstrap_central_test
from diffmm.eval.ranking import Ranking
from diffmm.utils.logging import get_logger

log = get_logger("ph5_hypothesis")


def _load_ranking(path: Path) -> Ranking:
    d = json.loads(path.read_text())
    return Ranking(tuple(d["agents"]), d["scores"], d["metric"])


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    ticker = cfg.data.ticker
    results_root = Path(cfg.cluster.results_root)

    pi_hist_path = results_root / "backtests" / "real" / ticker / "ranking_pi_hist.json"
    pi_diff_path = results_root / "backtests" / "synthetic" / ticker / "ranking_pi_diff.json"
    holdout_csv = results_root / "backtests" / "holdout" / ticker / "metrics.csv"

    for p, name in [(pi_hist_path, "π_hist"), (pi_diff_path, "π_diff"), (holdout_csv, "holdout metrics")]:
        if not p.exists():
            raise FileNotFoundError(f"{name} not found at {p}")

    pi_hist = _load_ranking(pi_hist_path)
    pi_diff = _load_ranking(pi_diff_path)
    holdout_rows = pl.read_csv(holdout_csv).to_dicts()

    log.info("π_hist : %s  (by %s)", pi_hist.agents, pi_hist.metric)
    log.info("π_diff : %s  (by %s)", pi_diff.agents, pi_diff.metric)

    n_boot = int(cfg.get("n_boot", 1000))
    seed = int(cfg.get("seed", 42))
    log.info("running bootstrap (n_boot=%d, seed=%d) on %d holdout rows",
             n_boot, seed, len(holdout_rows))
    result = bootstrap_central_test(holdout_rows, pi_hist, pi_diff, n_boot=n_boot, seed=seed)

    log.info("ρ(π_hist, π_truth) = %.3f  95%%CI = [%.3f, %.3f]",
             result.rho_hist_point, *result.rho_hist_ci)
    log.info("ρ(π_diff, π_truth) = %.3f  95%%CI = [%.3f, %.3f]",
             result.rho_diff_point, *result.rho_diff_ci)
    log.info("one-sided p-value (P[ρ_diff ≤ ρ_hist] over bootstrap) = %.4f",
             result.p_value_one_sided)
    if result.p_value_one_sided < 0.05:
        log.info("→ STATISTICALLY SIGNIFICANT support for H1 (ρ_diff > ρ_hist) at α=0.05")
    else:
        log.info("→ NOT statistically significant at α=0.05; report point estimates honestly")

    # Persist results
    out_dir = results_root / "hypothesis" / ticker
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "n_boot": result.n_boot,
        "rho_hist_point": result.rho_hist_point,
        "rho_diff_point": result.rho_diff_point,
        "rho_hist_ci_95": list(result.rho_hist_ci),
        "rho_diff_ci_95": list(result.rho_diff_ci),
        "p_value_one_sided": result.p_value_one_sided,
        "pi_hist": list(pi_hist.agents),
        "pi_diff": list(pi_diff.agents),
    }
    (out_dir / "bootstrap.json").write_text(json.dumps(summary, indent=2))
    np.savez(out_dir / "bootstrap_samples.npz",
             rho_hist=result.rho_hist_samples, rho_diff=result.rho_diff_samples)
    log.info("wrote %s", out_dir / "bootstrap.json")

    # Figure 7 — bar chart of point estimates with 95% CI error bars
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ["ρ(π_hist, π_truth)", "ρ(π_diff, π_truth)"]
    points = [result.rho_hist_point, result.rho_diff_point]
    errs = [
        [result.rho_hist_point - result.rho_hist_ci[0], result.rho_diff_point - result.rho_diff_ci[0]],
        [result.rho_hist_ci[1] - result.rho_hist_point, result.rho_diff_ci[1] - result.rho_diff_point],
    ]
    colors = ["#4a7ab8", "#b85a4a"]
    ax.bar(bars, points, yerr=errs, color=colors, capsize=8, edgecolor="black", linewidth=0.8)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylabel("Spearman ρ vs π_truth")
    ax.set_title(f"Central hypothesis test ({ticker})\n"
                 f"p = {result.p_value_one_sided:.3f}  (n_boot = {result.n_boot})")
    ax.set_ylim(-1.05, 1.05)
    plt.tight_layout()
    fig.savefig(out_dir / "figure7.pdf")
    fig.savefig(out_dir / "figure7.png", dpi=150)
    log.info("wrote %s and figure7.png", out_dir / "figure7.pdf")


if __name__ == "__main__":
    main()
