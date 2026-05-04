"""Phase 3 §3.2 — calibrate AS-family agents on a ticker's training tapes.

Writes one JSON per (ticker, strategy) under
results/agent_params/{ticker}_{strategy}.json. Each agent backtest later
loads its own JSON.

Usage:
    python scripts/20_calibrate_agents.py data=intc_top
    python scripts/20_calibrate_agents.py data=tsla_top \\
        +calibration.risk_aversion=0.05 +calibration.beta_ofi=2.0
"""
from __future__ import annotations

from pathlib import Path

import hydra
from omegaconf import DictConfig

from diffmm.agents.calibration import calibrate_a1, calibrate_a2, calibrate_a3, save_params
from diffmm.utils.logging import get_logger

log = get_logger("ph3_calibrate")


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig) -> None:
    train_dir = Path(cfg.data.processed_dir) / "train"
    train_paths = sorted(train_dir.glob(f"{cfg.data.ticker}_*.parquet"))
    if not train_paths:
        raise FileNotFoundError(f"no training tapes under {train_dir} for {cfg.data.ticker}")
    log.info("calibrating on %d training tapes for %s", len(train_paths), cfg.data.ticker)

    out_dir = Path(cfg.cluster.results_root) / "agent_params"
    out_dir.mkdir(parents=True, exist_ok=True)

    cal_cfg = cfg.get("calibration", {})
    γ = float(cal_cfg.get("risk_aversion", 0.1))
    base_size = int(cal_cfg.get("base_size", 100))
    inv_lim = int(cal_cfg.get("inventory_limit", 1000))
    tick = float(cal_cfg.get("tick_size_usd", cfg.data.tick_size_usd))
    β_ofi = float(cal_cfg.get("beta_ofi", 1.0))
    α_vpin = float(cal_cfg.get("alpha_vpin_widening", 1.0))
    τ_vpin = int(cal_cfg.get("tau_vpin_bucket", 2))
    kappa_override = cal_cfg.get("kappa_override", None)
    sigma_override = cal_cfg.get("sigma_override", None)
    if kappa_override is not None:
        kappa_override = float(kappa_override)
    if sigma_override is not None:
        sigma_override = float(sigma_override)

    common = dict(
        risk_aversion=γ, base_size=base_size,
        inventory_limit=inv_lim, tick_size_usd=tick,
        kappa_override=kappa_override, sigma_override=sigma_override,
    )
    # A0 (constant-spread) doesn't need σ/κ — it always quotes ±1 tick around
    # mid. Save its params separately with the same base_size + inventory limit
    # so the rest of the pipeline can treat it uniformly.
    from diffmm.agents.base import AgentParams
    a0 = AgentParams(base_size=base_size, inventory_limit=inv_lim)
    a1 = calibrate_a1(train_paths, **common)
    a2 = calibrate_a2(train_paths, **common, beta_ofi=β_ofi)
    a3 = calibrate_a3(train_paths, **common,
                      alpha_vpin_widening=α_vpin, tau_vpin_bucket=τ_vpin)

    log.info("A1: σ=%.3e  κ=%.3e  γ=%.3f", a1.volatility, a1.arrival_decay, a1.risk_aversion)

    for name, p in [("A0", a0), ("A1", a1), ("A2", a2), ("A3", a3)]:
        out_path = out_dir / f"{cfg.data.ticker}_{name}.json"
        save_params(p, out_path)
        log.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
