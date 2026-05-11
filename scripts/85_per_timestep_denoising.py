"""
85_per_timestep_denoising.py — Diagnostic showing WHERE in the denoising
process each architecture succeeds/fails. For a given trained checkpoint:

  1. Load real validation windows (256-event chunks).
  2. For each t ∈ {10, 50, 100, 200, 500, 800}:
       x_t = √ᾱ_t · x_0 + √(1-ᾱ_t) · ε,   ε ~ N(0, I)
  3. One-step denoise: x̂_0 = (x_t - √(1-ᾱ_t) · ε_θ(x_t, t)) / √ᾱ_t
  4. Compute per-feature reconstruction MSE between x̂_0 and x_0.

Outputs:
  results/per_timestep_<label>/
    per_timestep_mse.csv      (t, feature, mse)
    fig_per_timestep_mse.{pdf,png}

Run on cluster GPU node:
    sbatch --gres=gpu:1 --mem=32G --time=01:00:00 \\
        --wrap="python scripts/85_per_timestep_denoising.py \\
                --ckpt $HOME/orcd/scratch/diffmm/checkpoints/intc_v2/ckpt-best.ckpt \\
                --label v2 --real-dir data/processed/val \\
                --out results/per_timestep_v2"
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parent.parent

FEATURE_NAMES = [
    "bid_dist", "ask_dist", "bid_sz", "ask_sz",
    "trade_dist", "trade_sz", "trade_sign", "mid_return",
]
TIMESTEPS = (10, 50, 100, 200, 500, 800)


def load_real_windows(real_dir: Path, ticker: str, n_windows: int = 200,
                       window_size: int = 256, seed: int = 42) -> np.ndarray:
    """Load up to n_windows random 256-event windows from real validation data."""
    paths = sorted(real_dir.glob(f"{ticker}_*.parquet"))[:5]
    if not paths:
        sys.exit(f"FATAL: no real parquets at {real_dir}")
    rng = np.random.default_rng(seed)
    out = np.empty((n_windows, window_size, len(FEATURE_NAMES)), dtype=np.float32)
    found = 0
    for p in paths:
        df = pl.read_parquet(p).to_pandas()
        cols = [c for c in FEATURE_NAMES if c in df.columns]
        if len(cols) < len(FEATURE_NAMES):
            print(f"  WARN: missing cols in {p.name}: {set(FEATURE_NAMES) - set(cols)}")
            continue
        arr = df[FEATURE_NAMES].to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, copy=False)
        n_avail = arr.shape[0] - window_size
        if n_avail <= 0:
            continue
        n_take = min(n_windows - found, n_avail // 2)
        starts = rng.integers(0, n_avail, size=n_take)
        for s in starts:
            out[found] = arr[s:s + window_size]
            found += 1
            if found >= n_windows:
                return out
    return out[:found]


def cosine_alphas(T: int = 1000, s: float = 0.008) -> torch.Tensor:
    """Cosine schedule (Nichol & Dhariwal 2021). Returns ᾱ_t for t in [1, T]."""
    steps = torch.linspace(0, T, T + 1)
    f = torch.cos(((steps / T) + s) / (1 + s) * torch.pi / 2) ** 2
    alpha_bar = f / f[0]
    return alpha_bar.clamp(0.001, 0.9999)[1:]   # length T


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=Path, required=True)
    ap.add_argument("--label", required=True)
    ap.add_argument("--real-dir", type=Path,
                    default=ROOT / "data" / "processed" / "val")
    ap.add_argument("--ticker", default="INTC")
    ap.add_argument("--n-windows", type=int, default=200)
    ap.add_argument("--window-size", type=int, default=256)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="paper")

    # ── Load checkpoint via the project's loader ────────────────────────────
    # Defer import so the file syntax-checks even without the package
    from omegaconf import OmegaConf
    from diffmm.generator.model import GeneratorModel

    print(f"loading ckpt: {args.ckpt}")
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    cfg = state.get("hyper_parameters", {}).get("cfg", None)
    if cfg is None:
        sys.exit("ckpt missing hyper_parameters.cfg — pass --cfg-yaml manually")

    # Construct generator from config (FiLM vs AdaLN auto-detected)
    if cfg.generator.get("edm", {}).get("enabled", False):
        from diffmm.generator.edm import build_edm_generator
        gen = build_edm_generator(
            input_dim=8,
            window_length=args.window_size,
            conditioning_type=cfg.generator.model.conditioning_type,
            sigma_data=cfg.generator.edm.sigma_data,
            n_regimes=4,
        ).to(args.device)
    else:
        from diffmm.generator.trades_adapter import build_generator
        gen = build_generator(
            input_dim=8,
            window_length=args.window_size,
            conditioning_type=cfg.generator.model.conditioning_type,
            n_regimes=4,
        ).to(args.device)
    gen.load_state_dict({k.replace("generator.", "", 1): v
                         for k, v in state["state_dict"].items()
                         if k.startswith("generator.")}, strict=False)
    gen.eval()

    pred_type = cfg.generator.schedule.get("prediction_type", "eps")
    print(f"  arch: cond={cfg.generator.model.conditioning_type}  pred={pred_type}")

    # ── Load real windows ────────────────────────────────────────────────────
    X = load_real_windows(args.real_dir, args.ticker,
                            args.n_windows, args.window_size)
    print(f"loaded {X.shape[0]} real windows of shape {X.shape[1:]}")
    X_t = torch.from_numpy(X).to(args.device)

    # ── Cosine schedule for noising ──────────────────────────────────────────
    alpha_bar = cosine_alphas(T=1000).to(args.device)  # (1000,)
    rows: list[dict] = []

    with torch.no_grad():
        for t in TIMESTEPS:
            ab = alpha_bar[t - 1]  # 0-indexed
            sqrt_ab = torch.sqrt(ab)
            sqrt_1_ab = torch.sqrt(1 - ab)
            eps = torch.randn_like(X_t)
            x_t = sqrt_ab * X_t + sqrt_1_ab * eps

            # Use base regime (idx=0) for diagnostic — generator needs c_idx
            c_idx = torch.zeros(X_t.shape[0], dtype=torch.long, device=args.device)
            t_tensor = torch.full((X_t.shape[0],), t - 1,
                                   dtype=torch.long, device=args.device)
            try:
                pred = gen(x_t, t_tensor, c_idx)
            except Exception as e:
                print(f"  t={t}: forward failed — {e}")
                continue

            # Reconstruct x̂_0 from prediction
            if pred_type == "eps":
                x0_hat = (x_t - sqrt_1_ab * pred) / sqrt_ab
            elif pred_type == "v":
                # v = √ᾱ ε - √(1-ᾱ) x_0  →  x_0 = √ᾱ x_t - √(1-ᾱ) v
                x0_hat = sqrt_ab * x_t - sqrt_1_ab * pred
            else:
                x0_hat = pred  # x_0-prediction

            mse = ((x0_hat - X_t) ** 2).mean(dim=(0, 1)).cpu().numpy()
            for f, name in enumerate(FEATURE_NAMES):
                rows.append({"t": int(t), "feature": name,
                             "mse": float(mse[f])})
            print(f"  t={t}  total MSE = {mse.mean():.4g}")

    df = pd.DataFrame(rows)
    out_csv = args.out / "per_timestep_mse.csv"
    df.to_csv(out_csv, index=False)
    print(f"wrote {out_csv}")

    # ── Plot ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    pivot = df.pivot(index="t", columns="feature", values="mse")
    for feat in FEATURE_NAMES:
        if feat not in pivot.columns:
            continue
        ax.plot(pivot.index, pivot[feat], marker="o", linewidth=1.5,
                label=feat)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("noise step t (log)")
    ax.set_ylabel("per-feature reconstruction MSE (log)")
    ax.set_title(f"{args.label}: one-step denoising quality across noise levels")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    pdf = args.out / "fig_per_timestep_mse.pdf"
    png = args.out / "fig_per_timestep_mse.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")


if __name__ == "__main__":
    main()
