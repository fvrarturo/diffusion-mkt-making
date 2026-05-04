"""Run the entire (regime, seed) sampling manifest in a single Python process.

Avoids SLURM's MaxSubmitJobs QOS limit (64 on mit_normal_gpu) by processing
all manifest entries sequentially in one job. Each entry takes ~3-5 s of GPU
time, so 400 entries ≈ 25-40 min of wall-clock.

Loads the checkpoint ONCE at the start, then loops samples — much faster than
spawning a fresh Python process per entry via subprocess.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
from omegaconf import OmegaConf

from diffmm.data.dataset import N_FEATURES, NormStats
from diffmm.generator.decode import decode_batch_to_parquet, decode_stitched_batch_to_parquet
from diffmm.generator.sample import DDIMSchedule, ddim_sample
from diffmm.generator.trades_adapter import build_generator
from diffmm.io.manifest import write_meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True, help="path to manifest JSON")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--norm-stats", default="data/processed/norm_stats_INTC.json")
    ap.add_argument("--out-root", default=None,
                    help="synthetic root; defaults to $HOME/orcd/scratch/diffmm/synthetic")
    ap.add_argument("--ticker", default="INTC")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--window-length", type=int, default=256)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--n-steps", type=int, default=200)
    ap.add_argument("--guidance-weight", type=float, default=1.0)
    ap.add_argument("--x0-clip", type=float, default=4.0)
    ap.add_argument("--eta", type=float, default=0.0)
    ap.add_argument("--anchor-mid-override", type=float, default=None,
                    help="decode-time override of NormStats.anchor_mid (USD)")
    ap.add_argument("--n-windows-per-task", type=int, default=1,
                    help="Stitch this many sampled windows per (regime, seed) into a "
                         "single longer Parquet. Default 1 (no stitching). 10 → "
                         "2,560-event tapes (~4 min decoded wall-clock at 100ms spacing).")
    ap.add_argument("--prediction-type", default="eps", choices=["eps", "v"],
                    help="Must match the prediction_type the checkpoint was trained "
                         "with. v2/v3/v3.5 = 'eps' (default for backwards compat). "
                         "v4+ Phase B = 'v'. Mismatch produces garbage at sample time.")
    ap.add_argument("--conditioning-type", default="film", choices=["film", "adaln_zero"],
                    help="Must match the conditioning_type the checkpoint was trained "
                         "with. v2-v4 = 'film' (default). v5+ Phase C = 'adaln_zero'. "
                         "Mismatch causes load_state_dict to silently load nothing into "
                         "AdaLN parameters → samples will be garbage.")
    ap.add_argument("--start", type=int, default=0, help="start index into manifest")
    ap.add_argument("--end", type=int, default=None, help="end index (exclusive)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    regime_to_condition = {
        "base":     [0, 0, 1, 1],
        "high_vol": [2, 1, 1, 1],
        "toxic":    [2, 2, 1, 1],
        "thin":     [1, 2, 0, 1],
    }

    out_root = Path(args.out_root) if args.out_root else Path.home() / "orcd/scratch/diffmm/synthetic"
    # os.makedirs handles symlinked parents (e.g. ~/orcd/scratch) better than
    # Path.mkdir on Python 3.11.
    os.makedirs(out_root, exist_ok=True)

    manifest = json.loads(Path(args.manifest).read_text())
    end = args.end if args.end is not None else len(manifest)
    entries = manifest[args.start:end]
    print(f"[sample-batch] running {len(entries)} tasks ({args.start}..{end})", flush=True)

    # Load checkpoint once.
    print(f"[sample-batch] loading checkpoint {args.checkpoint} "
          f"(conditioning_type={args.conditioning_type})", flush=True)
    gen = build_generator(
        n_features=N_FEATURES,
        d_model=args.embed_dim * 2,
        num_heads=8,
        depth=8,
        max_seq_len=args.window_length + 8,
        embed_dim=args.embed_dim,
        num_diffusionsteps=1000,
        n_categories_per_axis=(3, 3, 3, 3),
        conditioning_type=args.conditioning_type,
    )
    state = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
    sd = {k.removeprefix("generator."): v
          for k, v in state["state_dict"].items()
          if k.startswith("generator.")}
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    if missing:
        print(f"[sample-batch] warning: missing keys {missing}", flush=True)
    gen.to(args.device).eval()

    schedule = DDIMSchedule.cosine(T=1000, prediction_type=args.prediction_type)
    print(f"[sample-batch] prediction_type={args.prediction_type}", flush=True)
    if not Path(args.norm_stats).exists():
        raise FileNotFoundError(
            f"norm stats {args.norm_stats} not found — required for decode (anchor_mid)."
        )
    norm = NormStats.load(args.norm_stats)
    if args.anchor_mid_override is not None:
        from dataclasses import replace
        print(f"[sample-batch] anchor_mid override: {args.anchor_mid_override} (was {norm.anchor_mid})", flush=True)
        norm = replace(norm, anchor_mid=float(args.anchor_mid_override))

    t_start = time.time()
    total_written = 0
    n_windows = max(1, int(args.n_windows_per_task))
    if n_windows > 1:
        print(f"[sample-batch] stitching {n_windows} windows per (regime, seed) → "
              f"{n_windows * args.window_length}-event tapes", flush=True)

    for i, entry in enumerate(entries):
        regime = entry["regime"]
        seed = int(entry["seed"])
        cond_vec = regime_to_condition[regime]

        cond = torch.tensor([cond_vec] * args.batch_size, dtype=torch.long, device=args.device)
        null = gen.regime_embed.null_condition(args.batch_size, args.device)

        if n_windows == 1:
            samples = ddim_sample(
                eps_theta=gen.predict_noise,
                shape=(args.batch_size, args.window_length, N_FEATURES),
                schedule=schedule,
                condition=cond,
                null_condition=null,
                guidance_weight=args.guidance_weight,
                n_steps=args.n_steps,
                device=args.device,
                seed=seed,
                eta=args.eta,
                x0_clip=args.x0_clip,
            )
            paths = decode_batch_to_parquet(
                samples, cond,
                out_dir=out_root,
                ticker=args.ticker,
                start_ts_ns=int(1_700_000_000_000_000_000) + seed * 86_400_000_000_000,
                seed=seed,
                regime_label=regime,
                norm_stats=norm,
            )
        else:
            # Sample N independent windows per batch element (different sub-seed
            # per window so noise is independent), stack into (N, B, L, F),
            # then decode-stitch each batch element.
            window_batches = []
            for k in range(n_windows):
                sub_seed = seed * 10_000 + k          # deterministic, distinct per window
                samples_k = ddim_sample(
                    eps_theta=gen.predict_noise,
                    shape=(args.batch_size, args.window_length, N_FEATURES),
                    schedule=schedule,
                    condition=cond,
                    null_condition=null,
                    guidance_weight=args.guidance_weight,
                    n_steps=args.n_steps,
                    device=args.device,
                    seed=sub_seed,
                    eta=args.eta,
                    x0_clip=args.x0_clip,
                )
                window_batches.append(samples_k)
            stacked = torch.stack(window_batches, dim=0)  # (N, B, L, F)
            paths = decode_stitched_batch_to_parquet(
                stacked, cond,
                out_dir=out_root,
                ticker=args.ticker,
                start_ts_ns=int(1_700_000_000_000_000_000) + seed * 86_400_000_000_000,
                seed=seed,
                regime_label=regime,
                norm_stats=norm,
            )
        total_written += len(paths)
        for p in paths:
            write_meta(
                p, seed=seed,
                creator_script="scripts/utils/run_sampling_manifest.py",
                input_artifacts=[args.checkpoint],
                extra={"regime": regime, "guidance_weight": args.guidance_weight,
                       "n_steps": args.n_steps, "x0_clip": args.x0_clip},
            )

        if (i + 1) % 20 == 0 or i + 1 == len(entries):
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 1e-6)
            print(
                f"[sample-batch] {i+1}/{len(entries)}  regime={regime}  seed={seed}  "
                f"elapsed={elapsed:.1f}s  rate={rate:.1f}/s  written={total_written}",
                flush=True,
            )

    print(f"[sample-batch] done. total files written: {total_written}", flush=True)


if __name__ == "__main__":
    main()
