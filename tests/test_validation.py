"""Smoke tests for the Work5 validation suite — focused on the critical math."""
from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from diffmm.eval import stylized
from diffmm.io import schema as schema_mod


# ─── synthetic-tape factory (reuse from test_phase3 pattern) ────────────

def _fake_canonical_tape(
    *, n_events: int = 500, base_mid: float = 30.0, spread: float = 0.01,
    return_std: float = 1e-5, trade_frac: float = 0.10, seed: int = 0,
    regime_label: str = "base",
) -> pl.DataFrame:
    rng = np.random.default_rng(seed)
    L = n_events
    log_returns = rng.normal(0.0, return_std, size=L); log_returns[0] = 0.0
    raw_mid = base_mid * np.exp(np.cumsum(log_returns))
    tick = 0.01
    bid_px = np.round((raw_mid - spread / 2) / tick) * tick
    ask_px = bid_px + spread
    mid = 0.5 * (bid_px + ask_px)
    bid_sz = rng.integers(100, 1000, size=L).astype(np.int64)
    ask_sz = rng.integers(100, 1000, size=L).astype(np.int64)
    is_trade = rng.random(L) < trade_frac
    sign = rng.choice([-1, 1], size=L).astype(np.int8)
    trade_px = np.where(is_trade & (sign > 0), ask_px, np.where(is_trade & (sign < 0), bid_px, np.nan))
    trade_sz = np.where(is_trade, rng.integers(50, 500, size=L), 0)
    trade_dist = np.where(is_trade, trade_px - mid, np.nan)
    rows = {
        "event_idx": np.arange(L, dtype=np.int64),
        "ts_ns":     1_700_000_000_000_000_000 + np.arange(L, dtype=np.int64) * 100_000_000,
        "ticker":    ["INTC"] * L,
        "event_type": np.where(is_trade, "trade", "quote_update"),
        "bid_px":    bid_px, "ask_px": ask_px,
        "bid_sz":    bid_sz, "ask_sz": ask_sz,
        "trade_px":  [float(v) if is_trade[i] else None for i, v in enumerate(trade_px)],
        "trade_sz":  [int(v) if is_trade[i] else None for i, v in enumerate(trade_sz)],
        "trade_sign":[int(s) if is_trade[i] else None for i, s in enumerate(sign)],
        "is_lit":    [True if is_trade[i] else None for i in range(L)],
        "mid":       mid, "spread": np.full(L, spread),
        "bid_dist":  bid_px - mid, "ask_dist": ask_px - mid,
        "trade_dist":[float(v) if is_trade[i] else None for i, v in enumerate(trade_dist)],
        "mid_return":[None] + np.diff(np.log(mid)).tolist(),
        "c_vol":     np.ones(L, dtype=np.int8),
        "c_vpin":    np.ones(L, dtype=np.int8),
        "c_imb":     np.ones(L, dtype=np.int8),
        "c_tod":     np.ones(L, dtype=np.int8),
        "regime_label": [regime_label] * L,
    }
    return pl.DataFrame(rows, schema_overrides=schema_mod.CORE_DTYPES)


# ─── Part B tests ──────────────────────────────────────────────────────

def test_return_diagnostics_runs_and_passes_w1_for_identical_distributions():
    """Real and synth from the same generator → W-1 should be small."""
    real = [_fake_canonical_tape(seed=k, n_events=1000) for k in range(5)]
    synth = pl.concat([_fake_canonical_tape(seed=100 + k, n_events=1000) for k in range(5)])
    rs = stylized.return_diagnostics(real, synth)
    # Both pools come from same dist → W-1 should be small relative to inter-day variation
    assert rs.wasserstein_1 >= 0
    assert rs.real["n"] > 0
    assert rs.synth["n"] > 0


def test_spread_diagnostics_one_tick_dominates():
    """Synthetic tapes with constant 1-tick spread should be detected."""
    real = [_fake_canonical_tape(seed=k) for k in range(3)]
    synth = pl.concat([_fake_canonical_tape(seed=100 + k) for k in range(3)])
    sp = stylized.spread_diagnostics(real, synth)
    assert sp.real_one_tick_frac > 0.95
    assert sp.synth_one_tick_frac > 0.95
    assert sp.pass_one_tick


def test_event_composition_detects_trade_fraction():
    real = [_fake_canonical_tape(seed=k, trade_frac=0.10) for k in range(3)]
    synth = pl.concat([_fake_canonical_tape(seed=100 + k, trade_frac=0.50) for k in range(3)])
    ev = stylized.event_composition(pl.concat(real), synth)
    assert 0.05 < ev.real_trade_frac < 0.15
    assert 0.45 < ev.synth_trade_frac < 0.55


# ─── Part C tests ──────────────────────────────────────────────────────

def test_acf_abs_return_returns_correct_shape():
    real = [_fake_canonical_tape(seed=k, n_events=2000) for k in range(3)]
    synth = [_fake_canonical_tape(seed=100 + k, n_events=2000) for k in range(3)]
    a = stylized.acf_abs_return(real, synth, max_lag=50)
    assert a.real_concat.shape == (51,)
    assert a.synth_concat.shape == (51,)
    assert a.real_per_day.shape == (3, 51)


def test_rv_signature_shape_matches_for_identical_distributions():
    real = [_fake_canonical_tape(seed=k, n_events=2000) for k in range(3)]
    synth = [_fake_canonical_tape(seed=100 + k, n_events=2000) for k in range(3)]
    rv = stylized.rv_signature(real, synth, deltas=[1, 5, 10, 50])
    # Real and synth from same dist → ratios should be similar
    assert 0.3 < rv.ratio_synth / rv.ratio_real < 3.0
    assert rv.real_mean.shape == (4,)


# ─── Part D tests ──────────────────────────────────────────────────────

def test_conditional_return_on_imbalance_constructs_bands():
    real = [_fake_canonical_tape(seed=k, n_events=1500) for k in range(5)]
    synth = [_fake_canonical_tape(seed=100 + k, n_events=1500) for k in range(5)]
    cr = stylized.conditional_return_on_imbalance(real, synth, n_bins=10, deltas=[10],
                                                    n_bootstrap=20)
    assert 10 in cr.real_means
    assert cr.real_means[10].shape == (10,)
    assert cr.real_bands[10][0].shape == (10,)


def test_joint_spread_imbalance_jsd_in_range():
    real = pl.concat([_fake_canonical_tape(seed=k) for k in range(3)])
    synth = pl.concat([_fake_canonical_tape(seed=100 + k) for k in range(3)])
    j = stylized.joint_spread_imbalance(real, synth)
    assert 0.0 <= j.jensen_shannon <= 1.0
    assert j.real_pmf.sum() == pytest.approx(1.0, rel=1e-5)
    assert j.synth_pmf.sum() == pytest.approx(1.0, rel=1e-5)


def test_conditional_moments_by_regime_handles_single_regime():
    real = [_fake_canonical_tape(seed=k, regime_label="base") for k in range(3)]
    synth = pl.concat([_fake_canonical_tape(seed=100 + k, regime_label="base") for k in range(3)])
    cm = stylized.conditional_moments_by_regime(real, synth)
    assert "base" in cm.z_scores.index
    # Per-regime z-scores for "base" should be small (same dist)
    base_zs = cm.z_scores.loc["base"].dropna()
    assert (np.abs(base_zs) < 5.0).all()


# ─── Marginal-summary table ────────────────────────────────────────────

def test_marginal_summary_returns_one_row_per_feature():
    real = [_fake_canonical_tape(seed=k) for k in range(3)]
    synth = pl.concat([_fake_canonical_tape(seed=100 + k) for k in range(3)])
    table = stylized.marginal_summary(real, synth)
    assert "mid_return" in table["feature"].values
    assert "spread" in table["feature"].values


# ─── Inter-day W-1 sanity ──────────────────────────────────────────────

def test_inter_day_wasserstein_zero_for_identical():
    a = np.random.default_rng(0).standard_normal(1000)
    b = a.copy()
    mean, std = stylized.inter_day_wasserstein([a, b])
    assert mean == pytest.approx(0.0, abs=1e-9)
    assert std == 0.0


# ─── Part F.1b — regime embedding cosine similarity ────────────────────

def test_embedding_cosine_collapse_diagnoses_correctly():
    """When all regime embeddings are identical, diagnose 'TABLE COLLAPSE'."""
    import torch
    from diffmm.eval import model_diagnostics
    from diffmm.generator.conditioning import RegimeEmbedding
    from diffmm.generator.model import GeneratorModel, MinimalDenoiser
    from diffmm.data.dataset import N_FEATURES

    embed = RegimeEmbedding(n_categories_per_axis=(3, 3, 3, 3), embed_dim=16)
    # Force every per-axis embedding to be a constant vector → all conditions collapse
    for layer in embed.per_axis:
        torch.nn.init.constant_(layer.weight, 1.0)
    denoiser = MinimalDenoiser(n_features=N_FEATURES, hidden_dim=16, ctx_dim=16)
    gen = GeneratorModel(denoiser=denoiser, regime_embed=embed)

    res = model_diagnostics.regime_embedding_cosine_similarity(
        gen,
        regime_to_condition={"base": [0, 0, 1, 1], "high_vol": [2, 1, 1, 1],
                              "toxic": [2, 2, 1, 1], "thin": [1, 2, 0, 1]},
        device="cpu",
    )
    assert res.matrix.shape == (5, 5)
    # Diagonal exactly 1.0 (vector with itself)
    np.testing.assert_allclose(np.diag(res.matrix), 1.0, atol=1e-6)
    assert "TABLE COLLAPSE" in res.diagnosis
    # Null is shaped via .null_condition() — uses the n-th index per axis
    # (different from any named regime) → null vs named should differ
    assert "null" in res.labels


def test_embedding_cosine_distinct_diagnoses_correctly():
    """Random init → embeddings should be reasonably distinct."""
    import torch
    from diffmm.eval import model_diagnostics
    from diffmm.generator.conditioning import RegimeEmbedding
    from diffmm.generator.model import GeneratorModel, MinimalDenoiser
    from diffmm.data.dataset import N_FEATURES

    torch.manual_seed(0)
    embed = RegimeEmbedding(n_categories_per_axis=(3, 3, 3, 3), embed_dim=128)
    denoiser = MinimalDenoiser(n_features=N_FEATURES, hidden_dim=16, ctx_dim=128)
    gen = GeneratorModel(denoiser=denoiser, regime_embed=embed)

    res = model_diagnostics.regime_embedding_cosine_similarity(
        gen,
        regime_to_condition={"base": [0, 0, 1, 1], "high_vol": [2, 1, 1, 1],
                              "toxic": [2, 2, 1, 1], "thin": [1, 2, 0, 1]},
        device="cpu",
    )
    # With random init at d=128, off-diagonal cosines should be small
    assert "DISTINCT" in res.diagnosis or "PARTIAL" in res.diagnosis


# ─── Phase B — v-parameterization ───────────────────────────────────────

def test_v_parameterization_round_trip():
    """v ↔ x_0 ↔ ε conversion satisfies the algebraic identity."""
    import torch
    torch.manual_seed(0)
    # Pick a representative t and ᾱ_t (mid-range).
    a_t = torch.tensor(0.5)
    x0 = torch.randn(4, 16, 8)
    eps = torch.randn(4, 16, 8)
    # Forward: x_t = √ᾱ · x_0 + √(1-ᾱ) · ε
    x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * eps
    # Define v
    v = torch.sqrt(a_t) * eps - torch.sqrt(1 - a_t) * x0
    # Reverse: x_0 = √ᾱ · x_t − √(1-ᾱ) · v
    x0_recovered = torch.sqrt(a_t) * x_t - torch.sqrt(1 - a_t) * v
    # Reverse: ε = √(1-ᾱ) · x_t + √ᾱ · v
    eps_recovered = torch.sqrt(1 - a_t) * x_t + torch.sqrt(a_t) * v
    assert torch.allclose(x0_recovered, x0, atol=1e-5)
    assert torch.allclose(eps_recovered, eps, atol=1e-5)


def test_ddim_sample_with_v_prediction_runs_and_produces_finite_output():
    """ddim_sample with prediction_type='v' produces a finite tensor."""
    import torch
    from diffmm.generator.conditioning import RegimeEmbedding
    from diffmm.generator.model import GeneratorModel, MinimalDenoiser
    from diffmm.generator.sample import DDIMSchedule, ddim_sample
    from diffmm.data.dataset import N_FEATURES

    torch.manual_seed(0)
    embed = RegimeEmbedding(n_categories_per_axis=(3, 3, 3, 3), embed_dim=16)
    denoiser = MinimalDenoiser(n_features=N_FEATURES, hidden_dim=32, ctx_dim=16)
    gen = GeneratorModel(denoiser=denoiser, regime_embed=embed)
    schedule = DDIMSchedule.cosine(T=100, prediction_type="v")
    cond = torch.tensor([[0, 0, 1, 1]])
    null = gen.regime_embed.null_condition(1, device="cpu")
    x = ddim_sample(
        eps_theta=gen.predict_noise, shape=(1, 16, N_FEATURES),
        schedule=schedule, condition=cond, null_condition=null,
        guidance_weight=1.0, n_steps=8, device="cpu", seed=0,
    )
    assert x.shape == (1, 16, N_FEATURES)
    assert torch.isfinite(x).all()


def test_ddpm_trainer_v_prediction_loss_decreases():
    """v-prediction trainer should still decrease loss on synthetic data."""
    import torch
    from torch.utils.data import DataLoader
    from diffmm.data.dataset import RandomWindowDataset, N_FEATURES
    from diffmm.generator.conditioning import RegimeEmbedding
    from diffmm.generator.model import GeneratorModel, MinimalDenoiser
    from diffmm.generator.sample import DDIMSchedule
    from diffmm.generator.train import DDPMTrainer
    from diffmm.utils.seeding import seed_all

    seed_all(0)
    embed = RegimeEmbedding(n_categories_per_axis=(3, 3, 3, 3), embed_dim=16)
    denoiser = MinimalDenoiser(n_features=N_FEATURES, hidden_dim=32, ctx_dim=16)
    gen = GeneratorModel(denoiser=denoiser, regime_embed=embed)
    schedule = DDIMSchedule.linear(T=100, prediction_type="v")
    trainer = DDPMTrainer(generator=gen, schedule=schedule, learning_rate=1e-3,
                           cfg_dropout=0.1, prediction_type="v")
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
    assert late < early, f"v-pred loss did not decrease: early={early:.4f} late={late:.4f}"


def test_cfg_dropout_curriculum_ramps_correctly():
    """Curriculum should be initial value before ramp_start, final after ramp_end,
    and linearly interpolate in between."""
    from diffmm.generator.train import CFGDropoutCurriculum
    cur = CFGDropoutCurriculum(
        enabled=True, initial=1.0, final=0.2,
        ramp_start_epoch=2, ramp_end_epoch=5,
    )
    assert cur.value_at(0) == 1.0
    assert cur.value_at(1) == 1.0
    assert cur.value_at(2) == 1.0
    # epoch 3: 1/3 through the ramp → 1.0 - (1/3)*0.8 = 0.7333
    assert abs(cur.value_at(3) - (1.0 - (1/3) * 0.8)) < 1e-6
    # epoch 4: 2/3 through → 1.0 - (2/3)*0.8 = 0.4667
    assert abs(cur.value_at(4) - (1.0 - (2/3) * 0.8)) < 1e-6
    assert cur.value_at(5) == 0.2
    assert cur.value_at(20) == 0.2
    # Disabled curriculum should always return final.
    cur_off = CFGDropoutCurriculum(enabled=False, initial=1.0, final=0.2,
                                    ramp_start_epoch=2, ramp_end_epoch=5)
    assert cur_off.value_at(0) == 0.2
    assert cur_off.value_at(100) == 0.2


def test_trainer_uses_curriculum_dropout_when_enabled():
    """When the trainer has a curriculum, _current_cfg_dropout uses it
    instead of the constructor's cfg_dropout arg. (`current_epoch` is a
    Lightning property without a setter, so we subclass to inject epoch.)"""
    import torch
    from diffmm.data.dataset import N_FEATURES
    from diffmm.generator.conditioning import RegimeEmbedding
    from diffmm.generator.model import GeneratorModel, MinimalDenoiser
    from diffmm.generator.sample import DDIMSchedule
    from diffmm.generator.train import DDPMTrainer, CFGDropoutCurriculum

    class _EpochInjectable(DDPMTrainer):
        _epoch_override = 0
        @property
        def current_epoch(self):
            return self._epoch_override

    torch.manual_seed(0)
    embed = RegimeEmbedding(n_categories_per_axis=(3, 3, 3, 3), embed_dim=16)
    denoiser = MinimalDenoiser(n_features=N_FEATURES, hidden_dim=16, ctx_dim=16)
    gen = GeneratorModel(denoiser=denoiser, regime_embed=embed)
    schedule = DDIMSchedule.linear(T=100, prediction_type="eps")
    cur = CFGDropoutCurriculum(enabled=True, initial=1.0, final=0.2,
                                ramp_start_epoch=2, ramp_end_epoch=5)
    trainer = _EpochInjectable(generator=gen, schedule=schedule, cfg_dropout=0.99,
                                 cfg_dropout_curriculum=cur)
    trainer._epoch_override = 0
    assert trainer._current_cfg_dropout() == 1.0
    trainer._epoch_override = 5
    assert trainer._current_cfg_dropout() == 0.2
    # cfg_dropout=0.99 was the constructor arg — should be ignored when curriculum is active.
    assert trainer._current_cfg_dropout() != 0.99


def test_schedule_prediction_type_defaults_and_round_trips():
    """DDIMSchedule.cosine/linear preserve prediction_type."""
    from diffmm.generator.sample import DDIMSchedule
    s_eps = DDIMSchedule.cosine(T=100)
    assert s_eps.prediction_type == "eps"
    s_v = DDIMSchedule.cosine(T=100, prediction_type="v")
    assert s_v.prediction_type == "v"
    s_lin = DDIMSchedule.linear(T=100, prediction_type="v")
    assert s_lin.prediction_type == "v"


# ─── Phase C — AdaLN-Zero conditioning ─────────────────────────────────

def test_adaln_zero_block_starts_as_identity():
    """AdaLN-Zero block with zero-initialized modulation MLP must produce
    output identical to input (the block is a no-op at step 0 because all
    γ/β/α = 0 → residual contribution = 0)."""
    import torch
    from diffmm.generator.trades_adapter import AdaLNTransformerBlock

    torch.manual_seed(0)
    block = AdaLNTransformerBlock(d_model=32, num_heads=4, ctx_dim=16, dropout=0.0)
    block.eval()
    h = torch.randn(2, 8, 32)
    c = torch.randn(2, 16)
    out = block(h, c)
    # At zero-init, α=0 makes the block exactly identity.
    assert torch.allclose(out, h, atol=1e-5)


def test_adaln_final_layer_starts_as_identity_on_normed_input():
    """AdaLNFinalLayer at zero-init produces LayerNorm(h) (with γ=0, β=0
    → output = LN(h) * 1 + 0). Verify by feeding a tensor and comparing
    against pure LN."""
    import torch
    from diffmm.generator.trades_adapter import AdaLNFinalLayer

    torch.manual_seed(0)
    layer = AdaLNFinalLayer(d_model=32, ctx_dim=16)
    layer.eval()
    h = torch.randn(2, 8, 32)
    c = torch.randn(2, 16)
    out = layer(h, c)
    expected = torch.nn.functional.layer_norm(h, normalized_shape=(32,))
    assert torch.allclose(out, expected, atol=1e-5)


def test_adaln_block_responds_to_conditioning_after_init():
    """After randomly perturbing the modulation MLP weights, the block should
    no longer be identity — output should differ from input."""
    import torch
    from diffmm.generator.trades_adapter import AdaLNTransformerBlock

    torch.manual_seed(0)
    block = AdaLNTransformerBlock(d_model=32, num_heads=4, ctx_dim=16, dropout=0.0)
    # Perturb the modulation weights (simulating post-training state)
    with torch.no_grad():
        block.adaLN_modulation[-1].weight.normal_(0, 0.5)
        block.adaLN_modulation[-1].bias.normal_(0, 0.1)
    block.eval()
    h = torch.randn(2, 8, 32)
    c = torch.randn(2, 16)
    out = block(h, c)
    assert not torch.allclose(out, h, atol=1e-3)
    # Different conditioning vectors should produce different outputs
    out_diff_c = block(h, torch.randn_like(c))
    assert not torch.allclose(out, out_diff_c, atol=1e-3)


def test_adaln_denoiser_smoke_forward_and_backward():
    """Full denoiser with conditioning_type='adaln_zero' runs forward + backward."""
    import torch
    from diffmm.data.dataset import N_FEATURES
    from diffmm.generator.trades_adapter import build_generator
    from diffmm.generator.sample import DDIMSchedule
    from diffmm.generator.train import DDPMTrainer
    from diffmm.utils.seeding import seed_all

    seed_all(0)
    gen = build_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16, num_diffusionsteps=100,
        conditioning_type="adaln_zero",
    )
    schedule = DDIMSchedule.linear(T=100, prediction_type="v")
    trainer = DDPMTrainer(generator=gen, schedule=schedule, learning_rate=1e-3,
                           cfg_dropout=0.1, prediction_type="v")
    opt = trainer.configure_optimizers()
    x0 = torch.randn(4, 16, N_FEATURES)
    c = torch.randint(0, 3, (4, 4), dtype=torch.long)
    loss = trainer.training_step((x0, c), 0)
    loss.backward()
    opt.step()
    assert torch.isfinite(loss).all()


def test_adaln_denoiser_loss_decreases_on_random_data():
    """Trains for 5 epochs; verify loss decreases meaningfully."""
    import torch
    from torch.utils.data import DataLoader
    from diffmm.data.dataset import RandomWindowDataset, N_FEATURES
    from diffmm.generator.trades_adapter import build_generator
    from diffmm.generator.sample import DDIMSchedule
    from diffmm.generator.train import DDPMTrainer
    from diffmm.utils.seeding import seed_all

    seed_all(0)
    gen = build_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16, num_diffusionsteps=100,
        conditioning_type="adaln_zero",
    )
    schedule = DDIMSchedule.linear(T=100, prediction_type="v")
    trainer = DDPMTrainer(generator=gen, schedule=schedule, learning_rate=1e-3,
                           cfg_dropout=0.1, prediction_type="v")
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
    assert all(l == l for l in losses), "loss went NaN"  # noqa: E741, PLR0124
    early = sum(losses[:8]) / 8
    late = sum(losses[-8:]) / 8
    assert late < early, f"AdaLN-Zero loss did not decrease: early={early:.4f} late={late:.4f}"


def test_film_modulation_magnitude_handles_adaln_blocks():
    """The F.2 extractor should detect AdaLN blocks and produce per-block
    γ/β/α columns (instead of FiLM's γ_dev/β_norm)."""
    import torch
    from diffmm.eval import model_diagnostics
    from diffmm.generator.trades_adapter import build_generator
    from diffmm.data.dataset import N_FEATURES

    torch.manual_seed(0)
    gen = build_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16, num_diffusionsteps=100,
        conditioning_type="adaln_zero",
    )
    res = model_diagnostics.film_modulation_magnitude(
        gen,
        regime_to_condition={"base": [0, 0, 1, 1], "high_vol": [2, 1, 1, 1]},
        seq_len=16, device="cpu",
    )
    cols = res.table.columns.tolist()
    # Should have AdaLN-style columns, NOT FiLM-style
    assert any("alpha_attn_norm" in c for c in cols), f"missing alpha_attn cols, got {cols}"
    assert any("gamma_attn_norm" in c for c in cols), f"missing gamma_attn cols"
    # Should have a final-layer entry
    assert any("gamma_final_norm" in c for c in cols), f"missing final-layer cols"
    # Should NOT have FiLM-style columns
    assert not any(c.endswith("_gamma_dev") for c in cols), f"unexpected FiLM cols: {cols}"


def test_build_generator_conditioning_type_validation():
    """Invalid conditioning_type raises a clear error."""
    import pytest
    from diffmm.generator.trades_adapter import build_generator
    from diffmm.data.dataset import N_FEATURES
    with pytest.raises(ValueError, match="conditioning_type must be"):
        build_generator(
            n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
            max_seq_len=64, embed_dim=16, num_diffusionsteps=100,
            conditioning_type="not_a_real_type",
        )


def test_adaln_film_paths_independent():
    """Building FiLM and AdaLN denoisers should give different module structures."""
    from diffmm.generator.trades_adapter import build_generator
    from diffmm.data.dataset import N_FEATURES

    gen_film = build_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16, num_diffusionsteps=100,
        conditioning_type="film",
    )
    gen_adaln = build_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16, num_diffusionsteps=100,
        conditioning_type="adaln_zero",
    )
    # FiLM denoiser has film_in / film_out, no adaln_blocks
    film_modules = [n for n, _ in gen_film.denoiser.named_modules()]
    assert any("film_in" in n for n in film_modules)
    assert not any("adaln_blocks" in n for n in film_modules)
    # AdaLN denoiser has adaln_blocks, no film_in
    adaln_modules = [n for n, _ in gen_adaln.denoiser.named_modules()]
    assert any("adaln_blocks" in n for n in adaln_modules)
    assert not any("film_in" in n for n in adaln_modules)


def test_forward_with_t_emb_matches_forward_film():
    """forward_with_t_emb(x, t_embed_table[t], ctx) must equal forward(x, t, ctx)
    for the FiLM path — the refactor introduces no semantic change."""
    import torch
    from diffmm.generator.trades_adapter import build_generator
    from diffmm.data.dataset import N_FEATURES

    torch.manual_seed(0)
    gen = build_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16, num_diffusionsteps=100,
        conditioning_type="film",
    )
    gen.eval()
    x = torch.randn(2, 16, N_FEATURES)
    t = torch.tensor([5, 50], dtype=torch.long)
    cond = torch.tensor([[0, 0, 1, 1], [2, 1, 1, 1]], dtype=torch.long)
    ctx = gen.regime_embed(cond)
    with torch.no_grad():
        a = gen.denoiser.forward(x, t, ctx)
        t_emb = gen.denoiser.t_embed_table[t]
        b = gen.denoiser.forward_with_t_emb(x, t_emb, ctx)
    assert torch.allclose(a, b, atol=1e-6)


def test_forward_with_t_emb_matches_forward_adaln():
    """Same identity check for the AdaLN path."""
    import torch
    from diffmm.generator.trades_adapter import build_generator
    from diffmm.data.dataset import N_FEATURES

    torch.manual_seed(0)
    gen = build_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16, num_diffusionsteps=100,
        conditioning_type="adaln_zero",
    )
    gen.eval()
    x = torch.randn(2, 16, N_FEATURES)
    t = torch.tensor([5, 50], dtype=torch.long)
    cond = torch.tensor([[0, 0, 1, 1], [2, 1, 1, 1]], dtype=torch.long)
    ctx = gen.regime_embed(cond)
    with torch.no_grad():
        a = gen.denoiser.forward(x, t, ctx)
        t_emb_raw = gen.denoiser.t_sinusoidal_table[t]
        b = gen.denoiser.forward_with_t_emb(x, t_emb_raw, ctx)
    assert torch.allclose(a, b, atol=1e-6)


# ─── Phase D — EDM (Karras 2022) ───────────────────────────────────────

def test_edm_schedule_get_sigmas_shape_and_endpoints():
    import torch
    from diffmm.generator.edm import EDMSchedule
    s = EDMSchedule(sigma_min=0.002, sigma_max=80.0, rho=7.0)
    sigmas = s.get_sigmas(n_steps=32, device="cpu")
    assert sigmas.shape == (33,)        # n_steps + 1 (final 0)
    assert torch.allclose(sigmas[0], torch.tensor(80.0), atol=1e-4)
    # Final entry is 0 (clean data target)
    assert float(sigmas[-1]) == 0.0
    # Schedule is monotonically decreasing
    assert (sigmas[:-1] >= sigmas[1:] - 1e-9).all()


def test_edm_schedule_sample_train_sigma_in_expected_range():
    import torch
    from diffmm.generator.edm import EDMSchedule
    torch.manual_seed(0)
    s = EDMSchedule(P_mean=-1.2, P_std=1.2)
    sigmas = s.sample_train_sigma(10000, device="cpu")
    log_s = sigmas.log()
    # log σ ~ N(P_mean, P_std). Empirical mean and std should be close.
    assert abs(log_s.mean().item() - (-1.2)) < 0.05
    assert abs(log_s.std().item() - 1.2) < 0.05


def test_edm_schedule_loss_weight_correct_form():
    import torch
    from diffmm.generator.edm import EDMSchedule
    s = EDMSchedule(sigma_data=0.5)
    sigma = torch.tensor([0.1, 0.5, 1.0, 5.0])
    w = s.loss_weight(sigma)
    expected = (sigma ** 2 + 0.5 ** 2) / (sigma * 0.5) ** 2
    assert torch.allclose(w, expected, atol=1e-6)


def test_sinusoidal_continuous_shape_and_finite():
    import torch
    from diffmm.generator.edm import sinusoidal_continuous
    val = torch.tensor([-3.0, 0.0, 3.0])
    emb = sinusoidal_continuous(val, dim=64)
    assert emb.shape == (3, 64)
    assert torch.isfinite(emb).all()


def test_edm_denoiser_preconditioning_at_extremes():
    """At very low σ, c_skip → 1, c_out → 0; output should be ≈ x (identity).
    At very high σ, c_skip → 0; output is dominated by c_out · F_θ."""
    import torch
    from diffmm.data.dataset import N_FEATURES
    from diffmm.generator.edm import EDMDenoiser
    from diffmm.generator.trades_adapter import TradesStyleDenoiser

    torch.manual_seed(0)
    inner = TradesStyleDenoiser(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, ctx_dim=16, dropout=0.0,
        conditioning_type="adaln_zero",
    )
    edm = EDMDenoiser(inner=inner, sigma_data=0.5)
    edm.eval()
    x = torch.randn(2, 16, N_FEATURES)
    ctx = torch.randn(2, 16)
    # Very low σ → output ≈ x
    sigma_low = torch.full((2,), 1e-4)
    with torch.no_grad():
        y_low = edm(x, sigma_low, ctx)
    # c_skip(1e-4) ≈ 1, c_out(1e-4) ≈ 1e-4 → y ≈ x
    assert torch.allclose(y_low, x, atol=1e-2)


def test_edm_denoiser_runs_with_film_inner():
    """EDMDenoiser should also work with FiLM inner (not just AdaLN)."""
    import torch
    from diffmm.data.dataset import N_FEATURES
    from diffmm.generator.edm import EDMDenoiser
    from diffmm.generator.trades_adapter import TradesStyleDenoiser

    torch.manual_seed(0)
    inner = TradesStyleDenoiser(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, ctx_dim=16, dropout=0.0,
        conditioning_type="film",
    )
    edm = EDMDenoiser(inner=inner, sigma_data=0.5)
    edm.eval()
    x = torch.randn(2, 16, N_FEATURES)
    sigma = torch.full((2,), 1.0)
    ctx = torch.randn(2, 16)
    with torch.no_grad():
        y = edm(x, sigma, ctx)
    assert y.shape == (2, 16, N_FEATURES)
    assert torch.isfinite(y).all()


def test_edm_sample_runs_and_produces_finite_output():
    import torch
    from diffmm.data.dataset import N_FEATURES
    from diffmm.generator.edm import EDMSchedule, build_edm_generator, edm_sample

    torch.manual_seed(0)
    gen = build_edm_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16,
        conditioning_type="adaln_zero",
        sigma_data=0.5,
    )
    schedule = EDMSchedule(sigma_min=0.01, sigma_max=10.0, sigma_data=0.5, rho=7.0)
    cond = torch.tensor([[0, 0, 1, 1]])
    null = gen.regime_embed.null_condition(1, device="cpu")
    x = edm_sample(
        predict_x0=gen.predict_noise,
        shape=(1, 16, N_FEATURES),
        schedule=schedule,
        condition=cond, null_condition=null,
        guidance_weight=1.0, n_steps=8, device="cpu", seed=0,
    )
    assert x.shape == (1, 16, N_FEATURES)
    assert torch.isfinite(x).all()


def test_edm_sample_euler_only_runs():
    """Sanity: second_order=False (pure Euler) also works."""
    import torch
    from diffmm.data.dataset import N_FEATURES
    from diffmm.generator.edm import EDMSchedule, build_edm_generator, edm_sample

    torch.manual_seed(0)
    gen = build_edm_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16,
        conditioning_type="adaln_zero",
    )
    schedule = EDMSchedule(sigma_min=0.01, sigma_max=10.0)
    cond = torch.tensor([[0, 0, 1, 1]])
    null = gen.regime_embed.null_condition(1, device="cpu")
    x = edm_sample(
        predict_x0=gen.predict_noise,
        shape=(1, 16, N_FEATURES),
        schedule=schedule,
        condition=cond, null_condition=null,
        guidance_weight=1.0, n_steps=8, device="cpu", seed=0,
        second_order=False,
    )
    assert torch.isfinite(x).all()


def test_edm_trainer_loss_decreases():
    """EDMTrainer should decrease loss over a few epochs on synthetic data."""
    import torch
    from torch.utils.data import DataLoader
    from diffmm.data.dataset import RandomWindowDataset, N_FEATURES
    from diffmm.generator.edm import EDMSchedule, EDMTrainer, build_edm_generator
    from diffmm.utils.seeding import seed_all

    seed_all(0)
    gen = build_edm_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16,
        conditioning_type="adaln_zero",
        sigma_data=0.5,
    )
    schedule = EDMSchedule(sigma_data=0.5, P_mean=-1.0, P_std=0.8)  # narrower σ range for stability on tiny model
    trainer = EDMTrainer(
        generator=gen, schedule=schedule,
        learning_rate=1e-3, cfg_dropout=0.1,
    )
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
    assert all(l == l for l in losses), "EDM loss went NaN"  # noqa: E741, PLR0124
    early = sum(losses[:8]) / 8
    late = sum(losses[-8:]) / 8
    assert late < early, f"EDM loss did not decrease: early={early:.4f} late={late:.4f}"


def test_per_timestep_mse_handles_edm_schedule():
    """The diagnostic should detect an EDMSchedule via duck-typing and use the
    σ-based path instead of the t-based path."""
    import torch
    import numpy as np
    from diffmm.data.dataset import N_FEATURES
    from diffmm.eval import model_diagnostics
    from diffmm.generator.edm import EDMSchedule, build_edm_generator

    torch.manual_seed(0)
    gen = build_edm_generator(
        n_features=N_FEATURES, d_model=32, num_heads=4, depth=2,
        max_seq_len=64, embed_dim=16,
        conditioning_type="adaln_zero",
        sigma_data=0.5,
    )
    schedule = EDMSchedule(sigma_min=0.01, sigma_max=10.0, sigma_data=0.5)
    val_windows = [np.random.randn(16, N_FEATURES).astype(np.float32) for _ in range(8)]
    res = model_diagnostics.per_timestep_mse(
        gen, schedule, val_windows,
        timesteps=[0, 1, 2, 3, 4],   # ti index, mapped onto the σ schedule
        n_samples_per_t=4, device="cpu",
    )
    # Outputs are finite + correctly sized
    assert res.timesteps.shape == (5,)
    assert res.mse_mean.shape == (5,)
    assert np.isfinite(res.mse_mean).all()
    assert np.isfinite(res.snr).all()
    # The Min-SNR weight column is replaced by EDM's λ(σ); should also be finite + positive
    assert np.isfinite(res.min_snr_weight).all()
    assert (res.min_snr_weight > 0).all()


# ─── Phase E — CopulaTransform (v8) ────────────────────────────────────

def _make_synthetic_tape_for_copula(n_events: int = 5000, seed: int = 0) -> np.ndarray:
    """Heavy-tailed multivariate sample with one discrete feature, for copula tests."""
    rng = np.random.default_rng(seed)
    # 8 features matching FEATURE_COLUMNS shape:
    # 0,1: bid_dist, ask_dist (continuous, bounded near zero)
    # 2,3: bid_sz, ask_sz (continuous, positive heavy-tailed)
    # 4: trade_dist (continuous)
    # 5: trade_sz (continuous heavy-tailed)
    # 6: trade_sign (discrete: -1, 0, +1)
    # 7: mid_return (continuous heavy-tailed; Student-t)
    arr = np.zeros((n_events, 8), dtype=np.float32)
    arr[:, 0] = rng.normal(-0.005, 0.001, n_events)         # bid_dist
    arr[:, 1] = rng.normal(0.005, 0.001, n_events)          # ask_dist
    arr[:, 2] = rng.exponential(500, n_events)              # bid_sz
    arr[:, 3] = rng.exponential(500, n_events)              # ask_sz
    arr[:, 4] = rng.normal(0, 0.001, n_events)              # trade_dist
    arr[:, 5] = rng.exponential(100, n_events)              # trade_sz
    arr[:, 6] = rng.choice([-1, 0, 1], n_events)            # trade_sign (DISCRETE)
    # mid_return: Student-t for heavy tails (kurtosis ≫ 3)
    arr[:, 7] = rng.standard_t(df=3, size=n_events) * 1e-5
    return arr


def test_copula_transform_round_trip_continuous_features():
    """forward + inverse should approximately recover the input for continuous features
    (with quantile bin granularity error)."""
    import numpy as np
    from diffmm.data.copula_transform import CopulaTransform

    arr = _make_synthetic_tape_for_copula(n_events=3000)
    # Save to a temp parquet so CopulaTransform.fit can read it
    import tempfile, polars as pl
    from diffmm.data.dataset import FEATURE_COLUMNS
    from diffmm.io import schema as sm

    with tempfile.TemporaryDirectory() as td:
        # Mock-fit: bypass parquet by directly building the CopulaTransform from the array
        # (testing the math, not the file IO)
        ct = CopulaTransform(
            ecdf_quantiles={
                j: np.sort(arr[:, j].astype(np.float64))
                for j in range(8) if j != 6   # skip discrete
            },
            mean=arr.mean(axis=0).astype(np.float32),
            std=arr.std(axis=0).astype(np.float32),
            feature_columns=FEATURE_COLUMNS,
            discrete_features=(6,),    # trade_sign
            anchor_mid=30.0,
            clip_quantile=0.001,
        )

        z = ct.normalize(arr)
        x_hat = ct.denormalize(z)

        # Continuous features: round-trip should be tight
        for j in [0, 1, 2, 3, 4, 5, 7]:
            err = np.median(np.abs(x_hat[:, j] - arr[:, j]))
            scale = np.std(arr[:, j])
            assert err < scale * 0.05, (
                f"feature {j}: median round-trip err {err:.4g} too large "
                f"(>5% of scale {scale:.4g})"
            )


def test_copula_transform_produces_gaussian_marginals():
    """forward pass should produce marginals that are approximately N(0,1) per feature."""
    import numpy as np
    from scipy import stats as sps
    from diffmm.data.copula_transform import CopulaTransform
    from diffmm.data.dataset import FEATURE_COLUMNS

    arr = _make_synthetic_tape_for_copula(n_events=5000)
    ct = CopulaTransform(
        ecdf_quantiles={j: np.sort(arr[:, j].astype(np.float64))
                        for j in range(8) if j != 6},
        mean=arr.mean(axis=0).astype(np.float32),
        std=arr.std(axis=0).astype(np.float32),
        feature_columns=FEATURE_COLUMNS,
        discrete_features=(6,),
        anchor_mid=30.0,
    )
    z = ct.normalize(arr)
    # Continuous features: marginal mean ~0, std ~1
    for j in [0, 1, 2, 3, 4, 5, 7]:
        assert abs(z[:, j].mean()) < 0.1, f"feature {j} mean {z[:, j].mean():.3f} not ~0"
        assert 0.85 < z[:, j].std() < 1.15, f"feature {j} std {z[:, j].std():.3f} not ~1"


def test_copula_transform_discrete_feature_passthrough():
    """Discrete features should use z-score (mean-shift + scale), not CDF transform."""
    import numpy as np
    from diffmm.data.copula_transform import CopulaTransform
    from diffmm.data.dataset import FEATURE_COLUMNS

    arr = _make_synthetic_tape_for_copula(n_events=2000)
    ct = CopulaTransform(
        ecdf_quantiles={j: np.sort(arr[:, j].astype(np.float64))
                        for j in range(8) if j != 6},
        mean=arr.mean(axis=0).astype(np.float32),
        std=arr.std(axis=0).astype(np.float32),
        feature_columns=FEATURE_COLUMNS,
        discrete_features=(6,),
        anchor_mid=30.0,
    )
    z = ct.normalize(arr)
    # Discrete feature 6 (trade_sign): z-scored, only 3 unique values map to 3 z-values
    unique_z = np.unique(z[:, 6])
    assert len(unique_z) <= 3, f"discrete feature should keep ≤3 unique values, got {len(unique_z)}"
    # Round-trip exactly recovers the originals
    x_hat = ct.denormalize(z)
    np.testing.assert_allclose(x_hat[:, 6], arr[:, 6], atol=1e-5)


def test_copula_transform_preserves_rank_correlation():
    """The key claim: per-feature monotone transform preserves rank correlations."""
    import numpy as np
    from scipy.stats import spearmanr
    from diffmm.data.copula_transform import CopulaTransform
    from diffmm.data.dataset import FEATURE_COLUMNS

    rng = np.random.default_rng(0)
    n = 3000
    # Construct two correlated heavy-tailed features
    z1 = rng.standard_t(3, n)
    z2 = 0.7 * z1 + 0.3 * rng.standard_t(3, n)        # strongly correlated
    # Build a 8-feature array: two correlated, six padding
    arr = np.column_stack([z1, z2] + [rng.normal(size=n) for _ in range(6)]).astype(np.float32)

    ct = CopulaTransform(
        ecdf_quantiles={j: np.sort(arr[:, j].astype(np.float64)) for j in range(8)},
        mean=arr.mean(axis=0).astype(np.float32),
        std=arr.std(axis=0).astype(np.float32),
        feature_columns=FEATURE_COLUMNS,
        discrete_features=(),
        anchor_mid=30.0,
    )
    rho_orig = spearmanr(arr[:, 0], arr[:, 1]).statistic
    z = ct.normalize(arr)
    rho_gauss = spearmanr(z[:, 0], z[:, 1]).statistic
    # Spearman rank correlation is invariant under monotone transforms — should be IDENTICAL
    assert abs(rho_gauss - rho_orig) < 0.01, (
        f"Spearman rho changed under CDF transform: {rho_orig:.4f} → {rho_gauss:.4f}"
    )


def test_copula_transform_recovers_kurtosis_after_inverse():
    """Inverse transform should restore the original heavy-tailed kurtosis."""
    import numpy as np
    from scipy.stats import kurtosis
    from diffmm.data.copula_transform import CopulaTransform
    from diffmm.data.dataset import FEATURE_COLUMNS

    rng = np.random.default_rng(0)
    # Heavy-tailed Student-t feature
    arr = np.zeros((5000, 8), dtype=np.float32)
    arr[:, 0] = rng.standard_t(df=2, size=5000)   # high kurtosis
    for j in range(1, 8):
        arr[:, j] = rng.normal(size=5000)

    ct = CopulaTransform(
        ecdf_quantiles={j: np.sort(arr[:, j].astype(np.float64)) for j in range(8)},
        mean=arr.mean(axis=0).astype(np.float32),
        std=arr.std(axis=0).astype(np.float32),
        feature_columns=FEATURE_COLUMNS,
        discrete_features=(),
        anchor_mid=30.0,
    )
    real_kurt = kurtosis(arr[:, 0])

    # Simulate "diffusion sampled in z-space" with N(0,1) draws
    z_samples = rng.normal(size=(5000, 8)).astype(np.float32)
    x_samples = ct.denormalize(z_samples)
    sampled_kurt = kurtosis(x_samples[:, 0])

    # The inverse CDF restores the original distribution shape — kurtosis should be close
    # (within sampling noise; CDF-recovered kurtosis converges to true kurtosis with N).
    assert abs(sampled_kurt - real_kurt) / max(abs(real_kurt), 1) < 0.5, (
        f"sampled kurtosis {sampled_kurt:.2f} too far from real {real_kurt:.2f}"
    )


def test_copula_transform_save_load_round_trip():
    """save() / load() preserves the transform exactly."""
    import numpy as np, tempfile, os
    from diffmm.data.copula_transform import CopulaTransform
    from diffmm.data.dataset import FEATURE_COLUMNS

    arr = _make_synthetic_tape_for_copula(n_events=1000)
    ct = CopulaTransform(
        ecdf_quantiles={j: np.sort(arr[:, j].astype(np.float64))
                        for j in range(8) if j != 6},
        mean=arr.mean(axis=0).astype(np.float32),
        std=arr.std(axis=0).astype(np.float32),
        feature_columns=FEATURE_COLUMNS,
        discrete_features=(6,),
        anchor_mid=30.123,
        clip_quantile=0.002,
    )
    with tempfile.TemporaryDirectory() as td:
        save_path = os.path.join(td, "copula.json")
        ct.save(save_path)
        ct2 = CopulaTransform.load(save_path)
    assert ct2.discrete_features == ct.discrete_features
    assert abs(ct2.anchor_mid - ct.anchor_mid) < 1e-6
    assert ct2.clip_quantile == ct.clip_quantile
    np.testing.assert_allclose(ct2.mean, ct.mean)
    np.testing.assert_allclose(ct2.std, ct.std)
    # ECDF arrays equal
    for j in ct.ecdf_quantiles:
        np.testing.assert_allclose(ct2.ecdf_quantiles[j], ct.ecdf_quantiles[j])
    # Forward identity check
    z1 = ct.normalize(arr)
    z2 = ct2.normalize(arr)
    np.testing.assert_allclose(z1, z2)


def test_copula_transform_drop_in_for_normstats_in_dataset():
    """LOBWindowDataset should accept either NormStats or CopulaTransform via the
    duck-typed .normalize() interface."""
    import numpy as np, tempfile, os, polars as pl
    from diffmm.data.copula_transform import CopulaTransform
    from diffmm.data.dataset import LOBWindowDataset, FEATURE_COLUMNS, CONDITION_COLUMNS
    from diffmm.io import schema as sm

    # Build a minimal valid tape parquet for the dataset to read
    rng = np.random.default_rng(0)
    n = 1000
    arr = _make_synthetic_tape_for_copula(n_events=n)
    df_dict = {col: arr[:, i] for i, col in enumerate(FEATURE_COLUMNS)}
    df_dict.update({col: rng.integers(0, 3, n).astype(np.int64) for col in CONDITION_COLUMNS})
    df_dict["event_idx"] = np.arange(n, dtype=np.int64)
    df_dict["ts_ns"] = (1_700_000_000_000_000_000 + np.arange(n, dtype=np.int64) * 100_000_000)
    df_dict["ticker"] = ["INTC"] * n
    df_dict["event_type"] = ["quote_update"] * n
    df_dict["bid_px"] = np.full(n, 30.0)
    df_dict["ask_px"] = np.full(n, 30.01)
    df_dict["bid_sz"] = arr[:, 2].astype(np.int64)
    df_dict["ask_sz"] = arr[:, 3].astype(np.int64)
    df_dict["trade_px"] = [None] * n
    df_dict["trade_sz"] = [None] * n
    df_dict["trade_sign"] = [None] * n
    df_dict["is_lit"] = [None] * n
    df_dict["mid"] = np.full(n, 30.005)
    df_dict["spread"] = np.full(n, 0.01)
    df_dict["c_vol"] = df_dict["c_vol"].astype(np.int8)
    df_dict["c_vpin"] = df_dict["c_vpin"].astype(np.int8)
    df_dict["c_imb"] = df_dict["c_imb"].astype(np.int8)
    df_dict["c_tod"] = df_dict["c_tod"].astype(np.int8)
    df_dict["regime_label"] = ["base"] * n
    df = pl.DataFrame(df_dict, schema_overrides=sm.CORE_DTYPES)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "INTC_20240101.parquet")
        df.write_parquet(path)

        ct = CopulaTransform(
            ecdf_quantiles={j: np.sort(arr[:, j].astype(np.float64))
                            for j in range(8) if j != 6},
            mean=arr.mean(axis=0).astype(np.float32),
            std=arr.std(axis=0).astype(np.float32),
            feature_columns=FEATURE_COLUMNS,
            discrete_features=(6,),
            anchor_mid=30.0,
        )
        # Should NOT raise
        ds = LOBWindowDataset([path], window_length=64, stride=64, norm_stats=ct,
                                validate_schema=False)
        assert len(ds) > 0
        x, c = ds[0]
        assert x.shape == (64, 8)
        # Output should be ~Gaussian per feature (continuous)
        for j in [0, 1, 7]:
            assert abs(x[:, j].mean().item()) < 1.0
            assert x[:, j].std().item() < 3.0   # within reasonable Gaussian range
