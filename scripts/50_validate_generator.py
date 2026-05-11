"""Generator validation suite (Work5).

Runs Parts B, C, D (statistics on real vs synthetic data) and Parts E, F
(model internals) per the Work5 plan. Writes ~20 figures + several CSVs to
`results/validation/<ticker>/`.

Compute split:
  - Parts B/C/D: CPU-only, no checkpoint loaded. Fast (~5-15 min).
  - Parts E/F:  needs checkpoint + 1 GPU. ~15-30 min.

Use `--parts` to select subsets. Default runs everything.

Usage:
  # CPU-only validation (no GPU needed)
  python scripts/50_validate_generator.py \\
      --parts B C D \\
      --real-dir data/processed/val/ \\
      --synth-dir /orcd/scratch/.../synthetic_v2_stitched/INTC/ \\
      --ticker INTC \\
      --out-dir results/validation/INTC/

  # GPU pass for model internals
  sbatch slurms/ph6_validate.slurm
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from diffmm.io import parquet
from diffmm.eval import stylized
from diffmm.viz import validation as viz
from diffmm.utils.logging import get_logger

log = get_logger("ph6_validate")


def _load_real(real_dir: Path, ticker: str) -> list[pl.DataFrame]:
    paths = sorted(real_dir.glob(f"{ticker}_*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no real val tapes under {real_dir} for {ticker}")
    log.info("loading %d real tapes from %s", len(paths), real_dir)
    return [parquet.read_tape(p) for p in paths]


def _sample_synth(synth_dir: Path, n_per_regime: int = 250, seed: int = 0,
                   regime_filter: str | None = None) -> list[pl.DataFrame]:
    """Load synthetic tapes from <synth_dir>/{base,high_vol,toxic,thin}/*.parquet.

    `regime_filter`: if set to one of {base, high_vol, toxic, thin}, look ONLY
    in that subdir. Falls back to treating `synth_dir` itself as the regime
    leaf if the subdir doesn't exist (used by per-regime evaluation).
    """
    paths_by_regime: dict[str, list[Path]] = {}
    candidates = [regime_filter] if regime_filter else ("base", "high_vol", "toxic", "thin")
    for r in candidates:
        rd = synth_dir / r if (synth_dir / r).exists() else synth_dir
        if rd.exists():
            paths = sorted(rd.glob("day_*.parquet")) or sorted(rd.glob("*.parquet"))
            if paths:
                paths_by_regime[r] = paths
    if not paths_by_regime:
        raise FileNotFoundError(
            f"no synthetic tapes under {synth_dir}"
            + (f" (regime_filter={regime_filter})" if regime_filter else "/{base,high_vol,toxic,thin}/")
        )
    rng = random.Random(seed)
    chosen: list[Path] = []
    for r, paths in paths_by_regime.items():
        chosen.extend(rng.sample(paths, min(n_per_regime, len(paths))))
    log.info("loading %d synth tapes (≤%d per regime%s)", len(chosen), n_per_regime,
             f", regime={regime_filter}" if regime_filter else "")
    return [parquet.read_tape(p) for p in chosen]


def _run_part_B(real_per_day: list[pl.DataFrame], synth_per_tape: list[pl.DataFrame],
                out_dir: Path) -> dict:
    log.info("=== Part B — Marginals ===")
    out = {}

    log.info("B.1 return distribution")
    rs = stylized.return_diagnostics(real_per_day, pl.concat(synth_per_tape, how="vertical_relaxed"))
    viz.plot_return_distribution(rs, out_dir / "B1_return_distribution")
    pd.DataFrame([{"key": k, **v} if isinstance(v, dict) else {"key": k, "val": v}
                  for k, v in {"real": rs.real, "synth": rs.synth,
                                "wasserstein_1": rs.wasserstein_1,
                                "inter_day_w1_mean": rs.inter_day_w1_mean,
                                "ks_stat": rs.ks_statistic,
                                "pass_w1": rs.pass_w1,
                                "pass_kurt": rs.pass_kurtosis}.items()
                  ]).to_csv(out_dir / "B1_return_stats.csv", index=False)
    out["return"] = rs

    log.info("B.2 spread distribution")
    sp = stylized.spread_diagnostics(real_per_day, pl.concat(synth_per_tape, how="vertical_relaxed"))
    viz.plot_spread_distribution(sp, out_dir / "B2_spread_distribution")
    pd.DataFrame([{"metric": k, "value": v} for k, v in {
        "real_one_tick_frac": sp.real_one_tick_frac,
        "synth_one_tick_frac": sp.synth_one_tick_frac,
        "wasserstein_1": sp.wasserstein_1,
        "inter_day_w1_mean": sp.inter_day_w1_mean,
        "chi_sq_stat": sp.chi_sq_statistic,
        "pass_one_tick": sp.pass_one_tick,
        "pass_w1": sp.pass_w1,
    }.items()]).to_csv(out_dir / "B2_spread_stats.csv", index=False)
    out["spread"] = sp

    log.info("B.3 size distribution")
    sz = stylized.size_diagnostics(real_per_day, pl.concat(synth_per_tape, how="vertical_relaxed"))
    viz.plot_size_distribution(sz, out_dir / "B3_size_distribution")
    out["size"] = sz

    log.info("B.4 event composition")
    real_concat = pl.concat(real_per_day, how="vertical_relaxed")
    synth_concat = pl.concat(synth_per_tape, how="vertical_relaxed")
    ev = stylized.event_composition(real_concat, synth_concat)
    viz.plot_event_composition(ev, out_dir / "B4_event_composition")
    pd.DataFrame([{"metric": k, "value": v} for k, v in {
        "real_trade_frac": ev.real_trade_frac,
        "synth_trade_frac": ev.synth_trade_frac,
        "real_buy_frac": ev.real_buy_frac,
        "synth_buy_frac": ev.synth_buy_frac,
        **{f"synth_trade_frac_{r}": v for r, v in ev.synth_trade_frac_by_regime.items()},
    }.items()]).to_csv(out_dir / "B4_event_stats.csv", index=False)
    out["event"] = ev

    log.info("B.5 marginal summary")
    ms = stylized.marginal_summary(real_per_day, synth_concat)
    ms.to_csv(out_dir / "B5_marginal_summary.csv", index=False)

    return out


def _run_part_C(real_per_day, synth_per_tape, out_dir: Path) -> dict:
    log.info("=== Part C — Temporal ===")
    out = {}
    log.info("C.1 ACF |return|")
    a1 = stylized.acf_abs_return(real_per_day, synth_per_tape)
    viz.plot_acf(a1, out_dir / "C1_acf_abs_return")
    out["acf_abs"] = a1

    log.info("C.2 ACF trade_sign")
    a2 = stylized.acf_trade_sign(real_per_day, synth_per_tape)
    viz.plot_acf(a2, out_dir / "C2_acf_trade_sign", include_loglog_inset=False)
    out["acf_sign"] = a2

    log.info("C.3 ACF return")
    a3 = stylized.acf_return(real_per_day, synth_per_tape)
    viz.plot_acf(a3, out_dir / "C3_acf_return", include_loglog_inset=False)
    out["acf_ret"] = a3

    log.info("C.4 RV signature")
    rv = stylized.rv_signature(real_per_day, synth_per_tape)
    viz.plot_rv_signature(rv, out_dir / "C4_rv_signature")
    out["rv"] = rv

    log.info("C.5 ACF OFI")
    a5 = stylized.acf_ofi(real_per_day, synth_per_tape)
    viz.plot_acf(a5, out_dir / "C5_acf_ofi", include_loglog_inset=False)
    out["acf_ofi"] = a5

    return out


def _run_part_D(real_per_day, synth_per_tape, out_dir: Path) -> dict:
    log.info("=== Part D — Conditional / Cross-feature ===")
    out = {}
    log.info("D.1 conditional E[r|I]")
    cr = stylized.conditional_return_on_imbalance(real_per_day, synth_per_tape)
    viz.plot_conditional_return_imbalance(cr, out_dir / "D1_conditional_return_imbalance")
    pd.DataFrame([
        {"delta": d, "beta_real": cr.betas_real[d], "beta_synth": cr.betas_synth[d],
         "r2_real": cr.r2_real[d], "r2_synth": cr.r2_synth[d],
         "pass_sign": cr.pass_sign_per_delta[d]}
        for d in cr.deltas
    ]).to_csv(out_dir / "D1_regression_stats.csv", index=False)
    out["cond_ret"] = cr

    log.info("D.2 joint spread-imbalance")
    real_concat = pl.concat(real_per_day, how="vertical_relaxed")
    synth_concat = pl.concat(synth_per_tape, how="vertical_relaxed")
    j = stylized.joint_spread_imbalance(real_concat, synth_concat)
    viz.plot_joint_spread_imbalance(j, out_dir / "D2_joint_spread_imbalance")
    out["joint"] = j

    log.info("D.3 conditional moments by regime")
    cm = stylized.conditional_moments_by_regime(real_per_day, synth_concat)
    viz.plot_conditional_moments(cm, out_dir / "D3_conditional_moments")
    cm.table.to_csv(out_dir / "D3_conditional_moments.csv", index=False)
    out["cond_mom"] = cm

    return out


def _load_normalizer(args):
    """Returns either NormStats or CopulaTransform based on --use-copula.
    Both expose .normalize() / .denormalize() / .anchor_mid duck-typed interface."""
    from diffmm.data.dataset import NormStats
    if args.use_copula:
        from diffmm.data.copula_transform import CopulaTransform
        copula_path = (
            Path(args.copula_path) if args.copula_path
            else Path(args.norm_stats).with_suffix(".copula.json")
        )
        log.info("loading CopulaTransform from %s", copula_path)
        return CopulaTransform.load(copula_path)
    return NormStats.load(args.norm_stats)


def _run_part_E(args, generator, schedule, real_per_day, out_dir: Path) -> dict:
    log.info("=== Part E — Training diagnostics ===")
    from diffmm.eval import model_diagnostics
    from diffmm.data.dataset import LOBWindowDataset, NormStats, FEATURE_COLUMNS

    # Convert real tapes to NORMALIZED (L, F) windows for model evaluation.
    # `.normalize()` dispatches through the duck-typed normalizer interface
    # so this works for both NormStats (z-score) and CopulaTransform (CDF).
    log.info("E.0 building val window pool")
    norm = _load_normalizer(args)
    val_windows = []
    for d in real_per_day:
        feats = d.select(FEATURE_COLUMNS).fill_null(0.0).to_numpy().astype(np.float32)
        feats_norm = norm.normalize(feats).astype(np.float32)
        n_chunks = feats_norm.shape[0] // 256
        for k in range(n_chunks):
            val_windows.append(feats_norm[k * 256: (k + 1) * 256])
    log.info("  %d val windows", len(val_windows))

    out = {}
    log.info("E.1 training curves")
    v1_log = model_diagnostics.parse_training_log(args.v1_log) if args.v1_log else pd.DataFrame()
    v2_log = model_diagnostics.parse_training_log(args.v2_log) if args.v2_log else pd.DataFrame()
    viz.plot_training_curves(v1_log, v2_log, out_dir / "E1_training_curves")

    log.info("E.2 per-timestep MSE (v2)")
    ptm_v2 = model_diagnostics.per_timestep_mse(generator, schedule, val_windows,
                                                 device=args.device)
    pd.DataFrame({
        "t": ptm_v2.timesteps, "mse_mean": ptm_v2.mse_mean,
        "mse_std": ptm_v2.mse_std, "snr": ptm_v2.snr,
        "min_snr_weight": ptm_v2.min_snr_weight,
    }).to_csv(out_dir / "E2_per_timestep_stats.csv", index=False)
    ptm_v1 = None
    if args.v1_checkpoint:
        log.info("E.2 per-timestep MSE (v1) — for comparison")
        try:
            from diffmm.generator.trades_adapter import build_generator
            from diffmm.data.dataset import N_FEATURES
            gen_v1 = build_generator(
                n_features=N_FEATURES, d_model=128, num_heads=8, depth=8,
                max_seq_len=264, embed_dim=64, num_diffusionsteps=1000,
                n_categories_per_axis=(3, 3, 3, 3),
            )
            import torch
            sd = torch.load(args.v1_checkpoint, map_location=args.device, weights_only=False)
            sd = {k.removeprefix("generator."): v for k, v in sd["state_dict"].items()
                  if k.startswith("generator.")}
            gen_v1.load_state_dict(sd, strict=False)
            gen_v1.to(args.device).eval()
            ptm_v1 = model_diagnostics.per_timestep_mse(gen_v1, schedule, val_windows,
                                                         device=args.device)
        except Exception as e:
            log.warning("v1 per-timestep load failed: %s", e)
    viz.plot_per_timestep_quality(ptm_v1, ptm_v2, out_dir / "E2_per_timestep_quality")
    out["ptm"] = ptm_v2

    log.info("E.3 x0_clip activation rate")
    norm = _load_normalizer(args)
    clip_v2 = model_diagnostics.x0_clip_activation(generator, schedule, norm,
                                                     n_seeds=args.n_seeds_clip,
                                                     device=args.device)
    viz.plot_x0_clip_activation(clip_v2, None, out_dir / "E3_x0_clip_activation")

    log.info("E.4 noise schedule")
    ns = model_diagnostics.noise_schedule_curves(schedule)
    viz.plot_noise_schedule(ns, out_dir / "E4_noise_schedule")

    return out


def _run_part_F(args, generator, schedule, real_per_day, out_dir: Path) -> dict:
    log.info("=== Part F — Model interpretability ===")
    from diffmm.eval import model_diagnostics
    from diffmm.data.dataset import NormStats, FEATURE_COLUMNS

    out = {}
    log.info("F.1 regime embedding PCA")
    pca = model_diagnostics.regime_embeddings_pca(generator)
    viz.plot_regime_embeddings(pca, out_dir / "F1_regime_embedding_pca")

    regime_to_condition = {
        "base":     [0, 0, 1, 1],
        "high_vol": [2, 1, 1, 1],
        "toxic":    [2, 2, 1, 1],
        "thin":     [1, 2, 0, 1],
    }

    log.info("F.1b regime embedding pairwise cosine similarity")
    cos = model_diagnostics.regime_embedding_cosine_similarity(
        generator, regime_to_condition, device=args.device,
    )
    viz.plot_regime_embedding_cosine(cos, out_dir / "F1b_embedding_cosine")
    pd.DataFrame(cos.matrix, index=cos.labels, columns=cos.labels).to_csv(
        out_dir / "F1b_embedding_cosine.csv")
    log.info("F.1b diagnosis: %s", cos.diagnosis)

    log.info("F.2 FiLM modulation magnitudes")
    film = model_diagnostics.film_modulation_magnitude(generator, regime_to_condition,
                                                          device=args.device)
    viz.plot_film_modulation(film, out_dir / "F2_film_modulation")
    film.table.to_csv(out_dir / "F2_film_modulation.csv", index=False)

    log.info("F.3 attention maps")
    norm = _load_normalizer(args)
    # Pick three real windows from each regime, take the first 256 events
    windows_by_label: dict[str, np.ndarray] = {}
    for label in ("base", "high_vol", "toxic"):
        for d in real_per_day:
            sub = d.filter(pl.col("regime_label") == label)
            if sub.height >= 256:
                feats = sub.select(FEATURE_COLUMNS).fill_null(0.0).to_numpy().astype(np.float32)
                feats_norm = norm.normalize(feats).astype(np.float32)
                windows_by_label[label] = feats_norm[:256]
                break
    if windows_by_label:
        att = model_diagnostics.attention_maps(generator, windows_by_label, device=args.device)
        viz.plot_attention_maps(att, out_dir / "F3_attention_maps")

    log.info("F.4 guidance sweep (this is the slow one — 50 seeds × 6 w values)")
    gs = model_diagnostics.guidance_sweep(
        generator, schedule, norm,
        regime_label="high_vol",
        regime_to_condition=regime_to_condition,
        real_per_day_for_target=real_per_day,
        n_seeds=args.n_seeds_sweep,
        device=args.device,
    )
    viz.plot_guidance_sweep(gs, out_dir / "F4_guidance_sweep")
    gs.table.to_csv(out_dir / "F4_guidance_sweep.csv", index=False)

    log.info("F.5 per-feature MSE")
    val_windows = []
    for d in real_per_day:
        feats = d.select(FEATURE_COLUMNS).fill_null(0.0).to_numpy().astype(np.float32)
        feats_norm = norm.normalize(feats).astype(np.float32)
        n_chunks = feats_norm.shape[0] // 256
        for k in range(n_chunks):
            val_windows.append(feats_norm[k * 256: (k + 1) * 256])
    pf = model_diagnostics.per_feature_mse(
        generator, schedule, val_windows,
        feature_names=list(FEATURE_COLUMNS),
        device=args.device,
    )
    viz.plot_per_feature_loss(pf, out_dir / "F5_per_feature_loss")
    pd.DataFrame({"feature": pf.feature_names, "mse": pf.mse_per_feature}).to_csv(
        out_dir / "F5_per_feature_loss.csv", index=False)

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["B", "C", "D", "E", "F"],
                    choices=["B", "C", "D", "E", "F"])
    ap.add_argument("--real-dir", required=True, type=Path)
    ap.add_argument("--synth-dir", required=True, type=Path)
    ap.add_argument("--ticker", default="INTC")
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--checkpoint", default=None,
                    help="v2 checkpoint path (required for parts E/F)")
    ap.add_argument("--v1-checkpoint", default=None)
    ap.add_argument("--norm-stats", default=None,
                    help="path to norm_stats_INTC.json (required for parts E/F)")
    ap.add_argument("--v1-log", default=None,
                    help="SLURM .out log for v1 training (for E.1)")
    ap.add_argument("--v2-log", default=None,
                    help="SLURM .out log for v2 training (for E.1)")
    ap.add_argument("--n-synth-per-regime", type=int, default=250)
    ap.add_argument("--regime", default=None,
                    choices=["base", "high_vol", "toxic", "thin"],
                    help="Restrict synth-tape loading to a single regime "
                         "(used by per-regime G1 evaluation in P0.D).")
    ap.add_argument("--n-seeds-clip", type=int, default=20)
    ap.add_argument("--n-seeds-sweep", type=int, default=20)
    ap.add_argument("--device", default=None)
    ap.add_argument("--prediction-type", default="eps", choices=["eps", "v"],
                    help="Must match the prediction_type the checkpoint was trained "
                         "with. v2/v3/v3.5 = 'eps' (default). v4+ Phase B = 'v'.")
    ap.add_argument("--conditioning-type", default="film", choices=["film", "adaln_zero"],
                    help="Must match the conditioning_type the checkpoint was trained "
                         "with. v2-v4 = 'film' (default). v5+ Phase C = 'adaln_zero'.")
    ap.add_argument("--use-edm", action="store_true",
                    help="Phase D / v6+: validate an EDM checkpoint. When set, "
                         "the model is built via build_edm_generator and Part E.2 "
                         "uses σ-based per-timestep MSE.")
    ap.add_argument("--edm-sigma-data", type=float, default=0.5)
    ap.add_argument("--use-copula", action="store_true",
                    help="Phase E / v8+: load CopulaTransform instead of NormStats. "
                         "Required for checkpoints trained with copula.enabled=true. "
                         "Part E and F functions still work because the model is built "
                         "the same way (only the normalizer differs).")
    ap.add_argument("--copula-path", default=None)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    t0 = time.time()
    real_per_day = _load_real(args.real_dir, args.ticker)
    synth_per_tape = _sample_synth(args.synth_dir, args.n_synth_per_regime,
                                    regime_filter=args.regime)
    log.info("data loaded in %.1fs", time.time() - t0)

    # Run requested parts
    all_results: dict = {}
    if "B" in args.parts:
        all_results.update(_run_part_B(real_per_day, synth_per_tape, args.out_dir))
    if "C" in args.parts:
        all_results.update(_run_part_C(real_per_day, synth_per_tape, args.out_dir))
    if "D" in args.parts:
        all_results.update(_run_part_D(real_per_day, synth_per_tape, args.out_dir))

    if any(p in args.parts for p in ("E", "F")):
        if not args.checkpoint or not args.norm_stats:
            log.error("Parts E/F require --checkpoint and --norm-stats")
            sys.exit(2)
        log.info(
            "loading checkpoint (prediction_type=%s, conditioning_type=%s, use_edm=%s)",
            args.prediction_type, args.conditioning_type, args.use_edm,
        )
        from diffmm.data.dataset import N_FEATURES
        import torch
        if args.use_edm:
            from diffmm.generator.edm import (
                EDMSchedule as _EDMSchedule, build_edm_generator,
            )
            generator = build_edm_generator(
                n_features=N_FEATURES, d_model=256, num_heads=8, depth=8,
                max_seq_len=264, embed_dim=128,
                n_categories_per_axis=(3, 3, 3, 3),
                conditioning_type=args.conditioning_type,
                sigma_data=args.edm_sigma_data,
            )
            schedule = _EDMSchedule(sigma_data=args.edm_sigma_data)
        else:
            from diffmm.generator.trades_adapter import build_generator
            from diffmm.generator.sample import DDIMSchedule
            generator = build_generator(
                n_features=N_FEATURES, d_model=256, num_heads=8, depth=8,
                max_seq_len=264, embed_dim=128, num_diffusionsteps=1000,
                n_categories_per_axis=(3, 3, 3, 3),
                conditioning_type=args.conditioning_type,
            )
            schedule = DDIMSchedule.cosine(T=1000, prediction_type=args.prediction_type)
        state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        sd = {k.removeprefix("generator."): v for k, v in state["state_dict"].items()
              if k.startswith("generator.")}
        generator.load_state_dict(sd, strict=False)
        generator.to(args.device).eval()

        if "E" in args.parts:
            all_results.update(_run_part_E(args, generator, schedule, real_per_day, args.out_dir))
        if "F" in args.parts:
            all_results.update(_run_part_F(args, generator, schedule, real_per_day, args.out_dir))

    # Consolidated pass/fail summary if all the right parts were run
    if all(k in all_results for k in ("return", "spread", "size", "event",
                                       "acf_abs", "acf_sign", "acf_ret", "rv",
                                       "cond_ret", "joint", "cond_mom")):
        log.info("=== G.1 Pass/Fail summary ===")
        pf = stylized.consolidated_summary(
            all_results["return"], all_results["spread"], all_results["size"],
            all_results["event"], all_results["acf_abs"], all_results["acf_sign"],
            all_results["acf_ret"], all_results["rv"], all_results["cond_ret"],
            all_results["joint"], all_results["cond_mom"],
        )
        pf.to_csv(args.out_dir / "G1_pass_fail_summary.csv", index=False)
        log.info("\n%s", pf.to_string(index=False))

    log.info("done in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
