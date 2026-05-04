"""Diagnostic: sweep (n_steps, guidance_weight) and print raw sampled-tensor
stats (pre-denormalization). Healthy DDIM output should be ~N(0, 1) in
normalized space. Wild values imply under-trained model, CFG overshoot, or
schedule/step-count mismatch.
"""
from __future__ import annotations

import argparse

import torch

from diffmm.data.dataset import N_FEATURES, NormStats
from diffmm.generator.sample import DDIMSchedule, ddim_sample
from diffmm.generator.trades_adapter import build_generator


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    gen = build_generator(
        n_features=N_FEATURES,
        d_model=args.embed_dim * 2,
        num_heads=8,
        depth=8,
        max_seq_len=args.seq_len + 8,
        embed_dim=args.embed_dim,
        num_diffusionsteps=1000,
        n_categories_per_axis=(3, 3, 3, 3),
    )
    state = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    sd = {k.removeprefix("generator."): v
          for k, v in state["state_dict"].items()
          if k.startswith("generator.")}
    missing, unexpected = gen.load_state_dict(sd, strict=False)
    if missing:
        print(f"[warn] missing keys: {missing}")
    if unexpected:
        print(f"[warn] unexpected keys: {unexpected}")
    gen.to(args.device).eval()

    schedule = DDIMSchedule.cosine(T=1000)
    B, L, F = args.batch_size, args.seq_len, N_FEATURES
    null = gen.regime_embed.null_condition(B, args.device)
    cond = torch.tensor([[0, 0, 1, 1]] * B, dtype=torch.long, device=args.device)

    print("raw sampled-tensor stats (normalized space; healthy = mean~0, std~1):")
    header = "  {:<8} {:<6} {:<10} {:<10} {:<10} {:<10}".format(
        "n_steps", "w", "mean", "std", "min", "max",
    )
    print(header)
    for n_steps in (20, 50, 200):
        for w in (0.0, 1.0, 3.0):
            x = ddim_sample(
                eps_theta=gen.predict_noise, shape=(B, L, F), schedule=schedule,
                condition=cond, null_condition=null,
                guidance_weight=w, n_steps=n_steps, device=args.device, seed=0,
            )
            print("  {:<8d} {:<6.1f} {:<10.3f} {:<10.3f} {:<10.3f} {:<10.3f}".format(
                n_steps, w, x.mean().item(), x.std().item(),
                x.min().item(), x.max().item(),
            ))

    # Per-channel stats at the best-looking config (n_steps=200, w=0)
    print()
    print("per-channel stats at n_steps=200, w=0 (normalized space, with x0_clip=4):")
    x = ddim_sample(
        eps_theta=gen.predict_noise, shape=(B, L, F), schedule=schedule,
        condition=cond, null_condition=null,
        guidance_weight=0.0, n_steps=200, device=args.device, seed=0,
    )
    from diffmm.data.dataset import FEATURE_COLUMNS
    for i, name in enumerate(FEATURE_COLUMNS):
        col = x[:, :, i]
        print("  {:<12} mean={:<+8.3f} std={:<7.3f} min={:<+8.3f} max={:<+8.3f}".format(
            name, col.mean().item(), col.std().item(),
            col.min().item(), col.max().item(),
        ))

    # Noise-prediction quality at different timesteps. Healthy: eps_pred stats
    # should match the true noise ~N(0, 1) regardless of t.
    print()
    print("noise-prediction stats by timestep (input x ~ N(0, 1); expect eps mean~0, std~1):")
    for t_val in (999, 900, 700, 500, 300, 100, 20):
        x_in = torch.randn(B, L, F, device=args.device)
        t_in = torch.full((B,), t_val, dtype=torch.long, device=args.device)
        eps = gen.predict_noise(x_in, t_in, cond)
        print("  t={:<4d}  eps: mean={:<+7.4f}  std={:<6.4f}  ||eps||^2/E[1]={:<6.3f}".format(
            t_val, eps.mean().item(), eps.std().item(),
            (eps ** 2).mean().item(),
        ))


if __name__ == "__main__":
    main()
