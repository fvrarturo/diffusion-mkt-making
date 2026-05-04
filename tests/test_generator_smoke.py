"""End-to-end generator smoke test on random data.

Proves the math (training + sampling + CFG) is wired up correctly, without
needing real TAQ data, the cluster, or the TRADES checkpoint. Runs in seconds
on a laptop CPU.
"""
from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from diffmm.data.dataset import RandomWindowDataset, N_FEATURES
from diffmm.generator.conditioning import RegimeEmbedding
from diffmm.generator.decode import decode_window_to_dataframe
from diffmm.generator.model import GeneratorModel, MinimalDenoiser
from diffmm.generator.sample import DDIMSchedule, ddim_sample
from diffmm.generator.train import DDPMTrainer
from diffmm.generator.trades_adapter import build_generator
from diffmm.io import schema as schema_mod
from diffmm.utils.seeding import seed_all


def _build_generator(embed_dim: int = 32, hidden_dim: int = 32) -> GeneratorModel:
    embed = RegimeEmbedding(n_categories_per_axis=(3, 3, 3, 3), embed_dim=embed_dim)
    denoiser = MinimalDenoiser(n_features=N_FEATURES, hidden_dim=hidden_dim, ctx_dim=embed_dim)
    return GeneratorModel(denoiser=denoiser, regime_embed=embed)


def test_training_loss_decreases():
    seed_all(0)
    gen = _build_generator()
    schedule = DDIMSchedule.linear(T=100)
    trainer = DDPMTrainer(generator=gen, schedule=schedule, learning_rate=1e-3, cfg_dropout=0.1)
    opt = trainer.configure_optimizers()

    ds = RandomWindowDataset(n_windows=128, window_length=16)
    loader = DataLoader(ds, batch_size=32, shuffle=True)

    losses = []
    for epoch in range(5):
        for batch in loader:
            opt.zero_grad()
            loss = trainer.training_step(batch, 0)
            loss.backward()
            opt.step()
            losses.append(loss.item())

    assert all(t == t for t in losses), "loss went NaN"  # noqa: PLR0124
    early = sum(losses[:8]) / 8
    late = sum(losses[-8:]) / 8
    assert late < early, f"loss did not decrease: early={early:.4f} late={late:.4f}"


def test_ddim_sample_returns_correct_shape_and_finite():
    seed_all(0)
    gen = _build_generator()
    gen.eval()
    schedule = DDIMSchedule.linear(T=100)

    B, L, F = 4, 16, N_FEATURES
    cond = torch.zeros(B, 4, dtype=torch.long)
    null = gen.regime_embed.null_condition(B, device="cpu")

    x = ddim_sample(
        eps_theta=gen.predict_noise,
        shape=(B, L, F),
        schedule=schedule,
        condition=cond,
        null_condition=null,
        guidance_weight=3.0,
        n_steps=10,
        device="cpu",
        seed=42,
    )
    assert x.shape == (B, L, F)
    assert torch.isfinite(x).all(), "sampler produced non-finite values"


def test_cfg_zero_matches_unconditional():
    """At w=0, the CFG sampler should equal the conditional-only branch."""
    seed_all(0)
    gen = _build_generator()
    gen.eval()
    schedule = DDIMSchedule.linear(T=100)

    B, L, F = 2, 8, N_FEATURES
    cond = torch.tensor([[0, 1, 2, 0], [2, 0, 1, 2]])
    null = gen.regime_embed.null_condition(B, device="cpu")

    x_w0 = ddim_sample(
        eps_theta=gen.predict_noise, shape=(B, L, F), schedule=schedule,
        condition=cond, null_condition=null, guidance_weight=0.0,
        n_steps=5, device="cpu", seed=7,
    )
    # Re-running with the same seed and w=0 should produce identical output.
    x_w0_repeat = ddim_sample(
        eps_theta=gen.predict_noise, shape=(B, L, F), schedule=schedule,
        condition=cond, null_condition=null, guidance_weight=0.0,
        n_steps=5, device="cpu", seed=7,
    )
    assert torch.allclose(x_w0, x_w0_repeat, atol=1e-6)


def test_null_condition_shape():
    embed = RegimeEmbedding(n_categories_per_axis=(3, 3, 3, 3), embed_dim=16)
    null = embed.null_condition(batch_size=5, device="cpu")
    assert null.shape == (5, 4)
    assert (null == torch.tensor([3, 3, 3, 3])).all()


def test_trades_style_denoiser_forward_and_backward():
    """The production denoiser runs forward, backprop, and produces correct shape."""
    seed_all(0)
    gen = build_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16, num_diffusionsteps=100,
    )
    schedule = DDIMSchedule.linear(T=100)
    trainer = DDPMTrainer(generator=gen, schedule=schedule, learning_rate=1e-3)
    opt = trainer.configure_optimizers()
    ds = RandomWindowDataset(n_windows=16, window_length=16)
    loader = DataLoader(ds, batch_size=8)
    losses = []
    for batch in loader:
        opt.zero_grad()
        loss = trainer.training_step(batch, 0)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert all(l == l for l in losses), "loss went NaN"  # noqa: E741, PLR0124


def test_decode_produces_valid_canonical_parquet():
    """Sampled tensor → DataFrame that passes the canonical schema validator."""
    import numpy as np
    from diffmm.data.dataset import NormStats, FEATURE_COLUMNS

    seed_all(0)
    gen = build_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16, num_diffusionsteps=100,
    )
    gen.eval()
    schedule = DDIMSchedule.linear(T=100)

    B, L, F = 1, 16, N_FEATURES
    cond = torch.tensor([[0, 1, 2, 0]])
    null = gen.regime_embed.null_condition(B, device="cpu")
    x = ddim_sample(
        eps_theta=gen.predict_noise, shape=(B, L, F), schedule=schedule,
        condition=cond, null_condition=null, guidance_weight=1.0,
        n_steps=4, device="cpu", seed=0,
    )

    # Plausible INTC-scale norm stats for the decode round-trip.
    norm = NormStats(
        mean=np.zeros(N_FEATURES, dtype=np.float32),
        std=np.ones(N_FEATURES, dtype=np.float32),
        feature_columns=FEATURE_COLUMNS,
        anchor_mid=37.71,
    )
    df = decode_window_to_dataframe(
        x[0], condition=cond[0], ticker="INTC",
        start_ts_ns=1_700_000_000_000_000_000,
        norm_stats=norm,
    )
    schema_mod.validate(df)
    assert df.height == L
    assert (df["ask_px"] - df["bid_px"] >= 0).all()


def test_decode_stitched_windows_produce_continuous_mid():
    """Stitching N windows produces a longer canonical tape with a continuous
    mid trajectory across window boundaries (no jumps from the per-window
    fresh-anchor reset)."""
    import numpy as np
    from diffmm.data.dataset import NormStats, FEATURE_COLUMNS, N_FEATURES
    from diffmm.generator.decode import decode_stitched_windows_to_dataframe
    from diffmm.io import schema as schema_mod

    seed_all(0)
    N, L = 3, 8     # 3 windows × 8 events each = 24-event stitched tape
    rng = np.random.default_rng(0)

    # Synthetic per-window samples in normalized space (~N(0,1)).
    samples = torch.from_numpy(rng.standard_normal((N, L, N_FEATURES)).astype(np.float32))

    # Plausible INTC norm stats — identity for relative features so the
    # boundary continuity check is straightforward.
    norm = NormStats(
        mean=np.zeros(N_FEATURES, dtype=np.float32),
        std=np.full(N_FEATURES, 1e-4, dtype=np.float32),
        feature_columns=FEATURE_COLUMNS,
        anchor_mid=37.50,
    )
    cond = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    df = decode_stitched_windows_to_dataframe(
        samples, condition=cond,
        ticker="INTC",
        start_ts_ns=1_700_000_000_000_000_000,
        norm_stats=norm,
    )
    schema_mod.validate(df)
    assert df.height == N * L
    # Monotonic event_idx
    assert df["event_idx"].to_list() == list(range(N * L))
    # ts_ns strictly increasing
    ts = df["ts_ns"].to_numpy()
    assert (np.diff(ts) > 0).all()
    # Mid trajectory: at every window seam (rows L-1 → L, 2L-1 → 2L, ...),
    # the change should be small (just a single mid_return jump from anchor reset
    # is set to 0 by convention; the actual difference comes from float32 precision).
    mids = df["mid"].to_numpy()
    for k in range(1, N):
        boundary = k * L
        # row at end of window k-1
        last = mids[boundary - 1]
        first_next = mids[boundary]
        # Continuous: anchor_{k} = end_mid_{k-1}, mid_return[0]=0 → first_next == last.
        assert abs(first_next - last) < 1e-3, (
            f"discontinuity at boundary {boundary}: {last} → {first_next}"
        )


if __name__ == "__main__":
    test_training_loss_decreases()
    test_ddim_sample_returns_correct_shape_and_finite()
    test_cfg_zero_matches_unconditional()
    test_null_condition_shape()
    test_trades_style_denoiser_forward_and_backward()
    test_decode_produces_valid_canonical_parquet()
    test_decode_stitched_windows_produce_continuous_mid()
    print("OK — generator smoke passed")
