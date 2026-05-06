"""Diffusion-model internals analyses.

Parts E (training diagnostics) + F (interpretability) of the Work5 plan.

Distinct from `stylized.py` because these need an instantiated checkpoint and
real GPU forward passes. Designed so each analysis can be run independently
and gracefully skipped (e.g. if v1 checkpoint is unavailable).
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
import torch
import torch.nn.functional as F


# ─── Part E.1 — Training loss curves ───────────────────────────────────

def parse_training_log(path: str | Path) -> pd.DataFrame:
    """Parse `[train] step N  loss X` lines from a SLURM .out file.

    Returns a DataFrame with columns ('step', 'loss'). If a CSVLogger
    metrics.csv exists at the conventional location, prefer that.
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=["step", "loss"])
    pat = re.compile(r"^\[train\]\s+step\s+(\d+)\s+loss\s+([\d.eE+-]+)")
    rows: list[tuple[int, float]] = []
    with open(path) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                rows.append((int(m.group(1)), float(m.group(2))))
    return pd.DataFrame(rows, columns=["step", "loss"])


def load_csv_logger(checkpoint_dir: str | Path) -> pd.DataFrame | None:
    """Look for a Lightning CSVLogger metrics.csv in the checkpoint dir tree."""
    root = Path(checkpoint_dir)
    candidates = list(root.rglob("metrics.csv"))
    if not candidates:
        return None
    return pd.read_csv(candidates[0])


# ─── Part E.2 — Per-timestep denoising quality ──────────────────────────

@dataclass
class PerTimestepResult:
    timesteps: np.ndarray
    mse_mean: np.ndarray
    mse_std: np.ndarray
    eps_pred_std: np.ndarray
    snr: np.ndarray
    min_snr_weight: np.ndarray         # min(SNR, γ) / SNR


def per_timestep_mse(
    generator,                                 # GeneratorModel
    schedule,                                  # DDIMSchedule (v2-v5) OR EDMSchedule (v6+)
    val_windows: list[np.ndarray],             # list of (L, F) numpy arrays in NORMALIZED space
    timesteps: list[int] = (10, 20, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 950, 999),
    n_samples_per_t: int = 200,
    min_snr_gamma: float = 5.0,
    device: str = "cuda",
    seed: int = 42,
    n_axes: tuple[int, ...] = (3, 3, 3, 3),
) -> PerTimestepResult:
    """Per-noise-level MSE diagnostic.

    For DDPM/DDIM checkpoints (v2-v5): for each integer timestep t in
    `timesteps`, sample n_samples random val windows, add the corresponding
    diffusion noise, predict the model's regression target (ε or v based on
    `schedule.prediction_type`), report per-sample MSE. Min-SNR weight
    matches the trainer's effective weighting.

    For EDM checkpoints (v6+, detected via duck-type on the schedule
    attributes): the same fixed integer "timesteps" are mapped onto a
    σ-spaced schedule via the EDMSchedule's get_sigmas, the model predicts
    x_0 directly (the EDM denoiser is x_0-parameterized), and the loss is
    EDM's σ-weighted MSE on x_0 with λ(σ) = (σ²+σ_data²)/(σ·σ_data)².
    """
    rng = np.random.default_rng(seed)
    is_edm = hasattr(schedule, "sigma_data") and hasattr(schedule, "get_sigmas")
    if is_edm:
        # Build a σ schedule of the same "size" as the requested timesteps so
        # the result tables are comparable in width across DDPM/EDM runs.
        sigmas_full = schedule.get_sigmas(len(timesteps), device)[:-1]   # drop the trailing 0
        a_bar = None
        pred_type = "edm"
    else:
        a_bar = schedule.alphas_cumprod.to(device)
        pred_type = getattr(schedule, "prediction_type", "eps")

    # Default condition vector: middle bucket on every axis (regime "base" enough)
    cond_vec = torch.tensor([0, 0, 1, 1], dtype=torch.long, device=device)

    if not val_windows:
        return PerTimestepResult(np.array(timesteps), np.zeros(len(timesteps)),
                                 np.zeros(len(timesteps)), np.zeros(len(timesteps)),
                                 np.zeros(len(timesteps)), np.zeros(len(timesteps)))

    # Stack all val windows, normalize sample selection
    L = val_windows[0].shape[0]
    stacked = np.stack([w[:L] for w in val_windows], axis=0)   # (n_windows, L, F)

    timesteps = list(timesteps)
    mse_means = np.zeros(len(timesteps))
    mse_stds = np.zeros(len(timesteps))
    pred_stds = np.zeros(len(timesteps))
    snr = np.zeros(len(timesteps))

    generator.eval()
    with torch.no_grad():
        for ti, t in enumerate(timesteps):
            # Sample n random val windows
            idx = rng.integers(0, stacked.shape[0], size=min(n_samples_per_t, stacked.shape[0]))
            x0 = torch.from_numpy(stacked[idx].astype(np.float32)).to(device)
            B = x0.shape[0]
            cond = cond_vec.unsqueeze(0).expand(B, -1).contiguous()
            eps_true = torch.randn_like(x0)

            if is_edm:
                # EDM path: σ-based noising, predict x_0 directly.
                sigma = sigmas_full[ti]
                sigma_b = sigma.expand(B)
                sigma_view = sigma.view(1, *([1] * (x0.dim() - 1)))
                x_t = x0 + sigma_view * eps_true
                pred = generator.predict_noise(x_t, sigma_b, cond)   # returns x_0_pred
                target = x0
                snr_t = float((schedule.sigma_data / sigma) ** 2)    # σ_data² / σ²
            else:
                t_tensor = torch.full((B,), int(t), dtype=torch.long, device=device)
                a_t = a_bar[t]
                x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * eps_true
                pred = generator.predict_noise(x_t, t_tensor, cond)
                if pred_type == "v":
                    target = torch.sqrt(a_t) * eps_true - torch.sqrt(1 - a_t) * x0
                else:
                    target = eps_true
                snr_t = float(a_t / (1 - a_t))

            mse = ((pred - target) ** 2).mean(dim=tuple(range(1, target.dim())))
            mse_means[ti] = float(mse.mean())
            mse_stds[ti] = float(mse.std())
            pred_stds[ti] = float(pred.std())
            snr[ti] = snr_t

    if is_edm:
        # EDM uses λ(σ) = (σ²+σ_data²)/(σ·σ_data)²; equivalently in SNR-space
        # with SNR = (σ_data/σ)², λ = (1 + 1/SNR) / σ_data² · σ². For the
        # diagnostic table we just report λ(σ) directly, expressed via SNR.
        sigmas_np = np.array([float(s) for s in sigmas_full.cpu()])
        sigma_data = schedule.sigma_data
        weight = (sigmas_np ** 2 + sigma_data ** 2) / (sigmas_np * sigma_data) ** 2
    elif pred_type == "v":
        # Match DDPMTrainer.v-prediction Min-SNR formula: weight / (SNR + 1).
        weight = np.minimum(snr, min_snr_gamma) / (snr + 1.0)
    else:
        weight = np.minimum(snr, min_snr_gamma) / np.maximum(snr, 1e-12)
    return PerTimestepResult(
        timesteps=np.array(timesteps),
        mse_mean=mse_means,
        mse_std=mse_stds,
        eps_pred_std=pred_stds,
        snr=snr,
        min_snr_weight=weight,
    )


# ─── Part E.3 — x0_clip activation rate ────────────────────────────────

@dataclass
class ClipActivationResult:
    per_step_rate: np.ndarray              # (n_steps,) fraction of |x0_pred| > clip
    overall_rate: float


def x0_clip_activation(
    generator,
    schedule,
    norm_stats,
    *,
    n_seeds: int = 50,
    n_steps: int = 200,
    x0_clip: float = 4.0,
    batch_size: int = 4,
    seq_len: int = 256,
    n_features: int = 8,
    device: str = "cuda",
) -> ClipActivationResult:
    """Run reverse DDIM, count how often |x0_pred| > x0_clip per step.

    Skipped for EDM checkpoints (they don't use x0_clip — preconditioning
    bounds outputs naturally). Returns an empty result so the caller can
    log the skip and move on.
    """
    if not hasattr(schedule, "alphas_cumprod"):
        return ClipActivationResult(
            per_step_rate=np.zeros(n_steps), overall_rate=0.0,
        )
    a_bar = schedule.alphas_cumprod.to(device)
    T = a_bar.shape[0]
    timesteps = torch.linspace(T - 1, 0, n_steps + 1, dtype=torch.long, device=device)
    cond_vec = torch.tensor([[0, 0, 1, 1]] * batch_size, dtype=torch.long, device=device)

    per_step_counts = np.zeros(n_steps)
    per_step_total = np.zeros(n_steps)

    generator.eval()
    with torch.no_grad():
        for s in range(n_seeds):
            gen_rng = torch.Generator(device=device).manual_seed(s)
            x = torch.randn((batch_size, seq_len, n_features), generator=gen_rng, device=device)
            for i in range(n_steps):
                t = timesteps[i]
                t_next = timesteps[i + 1]
                a_t = a_bar[t]
                a_next = a_bar[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)
                t_batch = t.expand(batch_size)
                eps = generator.predict_noise(x, t_batch, cond_vec)
                x0_pred = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t)
                count = int((x0_pred.abs() > x0_clip).sum())
                per_step_counts[i] += count
                per_step_total[i] += x0_pred.numel()
                x0_pred = x0_pred.clamp(-x0_clip, x0_clip)
                dir_xt = torch.sqrt(torch.clamp(1 - a_next, min=0.0)) * eps
                x = torch.sqrt(a_next) * x0_pred + dir_xt

    per_step_rate = per_step_counts / np.maximum(per_step_total, 1)
    overall = float(per_step_counts.sum() / max(per_step_total.sum(), 1))
    return ClipActivationResult(per_step_rate=per_step_rate, overall_rate=overall)


# ─── Part E.4 — Noise schedule ─────────────────────────────────────────

@dataclass
class NoiseScheduleResult:
    t: np.ndarray
    alpha_bar: np.ndarray
    sqrt_alpha_bar: np.ndarray
    sqrt_one_minus_alpha_bar: np.ndarray
    snr: np.ndarray


def noise_schedule_curves(schedule) -> NoiseScheduleResult:
    """For DDPM/DDIM: tabulate ᾱ_t and SNR(t) over the integer-t schedule.
    For EDM: tabulate σ_i and SNR(σ_i) = (σ_data/σ_i)² over the σ schedule
    (mapped onto a synthetic t-axis for plot compatibility).
    """
    if hasattr(schedule, "sigma_data") and hasattr(schedule, "get_sigmas"):
        # EDM path. Use the standard 32-step σ schedule for plotting.
        sigmas = schedule.get_sigmas(n_steps=32, device="cpu")[:-1].numpy()
        # Map σ → "virtual α_bar" for table compatibility:
        #   In EDM, SNR(σ) = σ_data² / σ². Define ᾱ-equivalent = SNR/(SNR+1).
        snr = (schedule.sigma_data / sigmas) ** 2
        alpha_bar = snr / (snr + 1.0)
        t = np.arange(len(alpha_bar))
        return NoiseScheduleResult(
            t=t,
            alpha_bar=alpha_bar,
            sqrt_alpha_bar=np.sqrt(alpha_bar),
            sqrt_one_minus_alpha_bar=np.sqrt(np.maximum(1 - alpha_bar, 0)),
            snr=snr,
        )
    a_bar = schedule.alphas_cumprod.cpu().numpy()
    t = np.arange(len(a_bar))
    return NoiseScheduleResult(
        t=t,
        alpha_bar=a_bar,
        sqrt_alpha_bar=np.sqrt(a_bar),
        sqrt_one_minus_alpha_bar=np.sqrt(np.maximum(1 - a_bar, 0)),
        snr=a_bar / np.maximum(1 - a_bar, 1e-12),
    )


# ─── Part F.1 — Regime embedding PCA ───────────────────────────────────

@dataclass
class EmbeddingPCAResult:
    points_2d: np.ndarray                 # (81+1, 2)
    labels: list[str]                     # length 82
    regime_labels: list[str]              # for coloring
    var_explained: np.ndarray             # (2,)


def regime_embeddings_pca(generator, n_axes_card: tuple[int, ...] = (3, 3, 3, 3)) -> EmbeddingPCAResult:
    """Extract per-tuple regime embeddings (concatenated per-axis) → PCA to 2D."""
    from sklearn.decomposition import PCA

    embed = generator.regime_embed
    device = next(generator.parameters()).device

    tuples = []
    embeddings = []
    for i0 in range(n_axes_card[0]):
        for i1 in range(n_axes_card[1]):
            for i2 in range(n_axes_card[2]):
                for i3 in range(n_axes_card[3]):
                    tuples.append((i0, i1, i2, i3))
                    c = torch.tensor([[i0, i1, i2, i3]], dtype=torch.long, device=device)
                    with torch.no_grad():
                        e = embed(c).squeeze(0).cpu().numpy()
                    embeddings.append(e)

    # Null embedding
    with torch.no_grad():
        null_e = embed(embed.null_condition(1, device)).squeeze(0).cpu().numpy()
    embeddings.append(null_e)
    tuples.append(("NULL",))

    X = np.stack(embeddings)
    pca = PCA(n_components=2)
    points_2d = pca.fit_transform(X)

    # Map (c_vol, c_vpin) → regime_label
    def _label(t):
        if t == ("NULL",):
            return "null"
        cv, cp = t[0], t[1]
        if cv == 2 and cp == 2: return "toxic"
        if cv == 2: return "high_vol"
        if cp == 2: return "thin"
        return "base"

    labels = [str(t) for t in tuples]
    regime_labels = [_label(t) for t in tuples]

    return EmbeddingPCAResult(
        points_2d=points_2d,
        labels=labels,
        regime_labels=regime_labels,
        var_explained=pca.explained_variance_ratio_,
    )


# ─── Part F.1b — Regime-embedding cosine similarity ────────────────────

@dataclass
class EmbeddingCosineResult:
    labels: list[str]                     # row/col order, e.g. ["base", "high_vol", "toxic", "thin", "null"]
    matrix: np.ndarray                    # (k, k) cosine similarities
    diagnosis: str                        # one-line interpretation


def regime_embedding_cosine_similarity(
    generator,
    regime_to_condition: dict[str, list[int]],
    device: str = "cuda",
    *,
    table_collapse_threshold: float = 0.99,
    layer_collapse_threshold: float = 0.90,
) -> EmbeddingCosineResult:
    """Pairwise cosine similarity between named-regime context vectors and null.

    Diagnoses the FiLM-collapse failure mode (Work5 §F.2):
      • If all named-regime pairs cosine > table_collapse_threshold → the
        embedding *table itself* has collapsed (RegimeEmbedding learned the
        same vector for every category). Fix: rework conditioning input
        (larger embed_dim, contrastive aux loss, conditioning curriculum).
      • If named-regime pairs cosine < layer_collapse_threshold but downstream
        FiLM γ/β still match across regimes (per F.2) → the *FiLM layer*
        ignores the embedding. Fix: train FiLM longer, increase pdrop on CFG,
        depth-aware FiLM init.
      • If named-regime pairs cosine ∈ [layer_collapse_threshold,
        table_collapse_threshold] → mixed problem; both fixes likely needed.
    """
    embed = generator.regime_embed
    vecs: dict[str, np.ndarray] = {}
    with torch.no_grad():
        for name, cond_vec in regime_to_condition.items():
            c = torch.tensor([cond_vec], dtype=torch.long, device=device)
            v = embed(c).squeeze(0).cpu().numpy()
            vecs[name] = v
        null_c = embed.null_condition(1, device)
        vecs["null"] = embed(null_c).squeeze(0).cpu().numpy()

    labels = list(vecs.keys())
    k = len(labels)
    M = np.zeros((k, k), dtype=np.float64)
    for i, a in enumerate(labels):
        for j, b in enumerate(labels):
            va, vb = vecs[a], vecs[b]
            M[i, j] = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-12))

    # Off-diagonal stats over the named-regime block (exclude null row/col).
    n_named = k - 1 if labels[-1] == "null" else k
    off = []
    for i in range(n_named):
        for j in range(n_named):
            if i != j:
                off.append(M[i, j])
    off = np.asarray(off)
    null_pair_max = float(M[:n_named, n_named].max()) if labels[-1] == "null" else float("nan")

    if off.size == 0:
        diagnosis = "only one regime — diagnostic not applicable"
    elif off.min() > table_collapse_threshold:
        diagnosis = (
            f"EMBEDDING TABLE COLLAPSE: all named-regime pairs cosine ≥ "
            f"{off.min():.3f} (> {table_collapse_threshold}). The "
            f"RegimeEmbedding has learned ~the same vector for every regime. "
            f"Fix: enlarge embed_dim, add contrastive aux loss, conditioning "
            f"curriculum. (null vs named max cosine = {null_pair_max:.3f}.)"
        )
    elif off.max() < layer_collapse_threshold:
        diagnosis = (
            f"EMBEDDINGS DISTINCT (max pairwise cosine {off.max():.3f} < "
            f"{layer_collapse_threshold}). Embedding table is fine; check "
            f"the downstream conditioning layer (FiLM γ/β or AdaLN γ/β/α "
            f"per-block magnitudes via F.2) to see if modulation actually "
            f"differs across regimes. If F.2 std/mean across regimes is "
            f"small, conditioning is collapsing downstream — fix with "
            f"longer training, larger conditioning capacity, or AdaLN-Zero."
        )
    else:
        diagnosis = (
            f"PARTIAL COLLAPSE: named-regime pairwise cosine ∈ "
            f"[{off.min():.3f}, {off.max():.3f}]. Embeddings are similar but "
            f"not identical. Both embedding-side and FiLM-side fixes "
            f"recommended. (null vs named max cosine = {null_pair_max:.3f}.)"
        )

    return EmbeddingCosineResult(labels=labels, matrix=M, diagnosis=diagnosis)


# ─── Part F.2 — FiLM modulation magnitudes ─────────────────────────────

@dataclass
class FiLMResult:
    table: pd.DataFrame                   # rows = regime, cols = γ/β magnitudes per layer


def film_modulation_magnitude(
    generator,
    regime_to_condition: dict[str, list[int]],
    seq_len: int = 256,
    d_model: int = 256,
    device: str = "cuda",
) -> FiLMResult:
    """For each regime + null, capture conditioning modulation magnitudes at
    every modulating layer in the denoiser.

    Two paths supported:
      - **FiLM denoiser** (v2-v4): hooks `FiLMLayer.proj` outputs.
        Each FiLM layer produces (γ, β); we report ||γ|| (γ is 0-init so
        deviation ≡ norm) and ||β||.
      - **AdaLN-Zero denoiser** (v5+): hooks the per-block `adaLN_modulation`
        Linear outputs. Each AdaLN block produces 6 chunks
        (γ_attn, β_attn, α_attn, γ_mlp, β_mlp, α_mlp). The final layer produces
        2 (γ_final, β_final). For each, we report the 2-norm; AdaLN params
        are also 0-init so deviation ≡ norm.

    The function name is preserved for backwards compat with the validation
    pipeline. Returns the same dataclass; column names depend on the
    conditioning type detected.
    """
    captured: dict[str, list[torch.Tensor]] = {}

    def make_hook(name):
        def fn(module, input, output):
            captured.setdefault(name, []).append(output.detach().cpu())
        return fn

    # Detect which conditioning architecture is in use, and register hooks
    # on the appropriate modulation-producing layers.
    handles = []
    is_adaln = False
    for attr_name, module in generator.denoiser.named_modules():
        cls_name = module.__class__.__name__
        if cls_name == "FiLMLayer":
            handles.append(module.proj.register_forward_hook(make_hook(attr_name)))
        elif cls_name in ("AdaLNTransformerBlock", "AdaLNFinalLayer"):
            # The Sequential's last Linear is what produces the modulation params.
            # Hook it to capture (γ, β, α, ...) outputs in raw form.
            is_adaln = True
            mod_proj = module.adaLN_modulation[-1]
            handles.append(mod_proj.register_forward_hook(make_hook(attr_name)))

    def _row_from_capture(regime: str) -> dict:
        row: dict = {"regime": regime}
        for name, outputs in captured.items():
            tensor = outputs[0][0]       # (B=1, n_chunks * d_model) — first sample
            if is_adaln and "final" not in name:
                # AdaLN block: 6 chunks
                g1, b1, a1, g2, b2, a2 = tensor.chunk(6, dim=-1)
                row[f"{name}_gamma_attn_norm"] = float(g1.norm())
                row[f"{name}_beta_attn_norm"] = float(b1.norm())
                row[f"{name}_alpha_attn_norm"] = float(a1.norm())
                row[f"{name}_gamma_mlp_norm"] = float(g2.norm())
                row[f"{name}_beta_mlp_norm"] = float(b2.norm())
                row[f"{name}_alpha_mlp_norm"] = float(a2.norm())
            elif is_adaln:
                # AdaLN final layer: 2 chunks
                g, b = tensor.chunk(2, dim=-1)
                row[f"{name}_gamma_final_norm"] = float(g.norm())
                row[f"{name}_beta_final_norm"] = float(b.norm())
            else:
                # FiLM: 2 chunks (γ, β)
                gamma, beta = tensor.chunk(2, dim=-1)
                row[f"{name}_gamma_dev"] = float(gamma.norm())
                row[f"{name}_beta_norm"] = float(beta.norm())
        return row

    # EDM-wrapped denoisers have a `.inner` attribute (the underlying TRADES
    # denoiser). For EDM, `predict_noise`'s second arg is interpreted as σ,
    # not as integer timestep — and σ=0 sends c_noise = log(0)/4 → −∞,
    # corrupting all downstream activations into NaN. Detect EDM and pass
    # σ ≈ 0.3 (the median training σ from log-normal P_mean=-1.2) instead.
    is_edm = hasattr(generator.denoiser, "inner")

    def _make_t_arg() -> torch.Tensor:
        if is_edm:
            return torch.full((1,), 0.30, dtype=torch.float32, device=device)
        return torch.zeros(1, dtype=torch.long, device=device)

    rows = []
    try:
        for regime, cond_vec in regime_to_condition.items():
            captured.clear()
            cond = torch.tensor([cond_vec], dtype=torch.long, device=device)
            x = torch.zeros((1, seq_len, generator.denoiser.n_features), device=device)
            t = _make_t_arg()
            with torch.no_grad():
                generator.predict_noise(x, t, cond)
            rows.append(_row_from_capture(regime))
        # Null
        captured.clear()
        cond_null = generator.regime_embed.null_condition(1, device)
        x = torch.zeros((1, seq_len, generator.denoiser.n_features), device=device)
        with torch.no_grad():
            generator.predict_noise(x, _make_t_arg(), cond_null)
        rows.append(_row_from_capture("null"))
    finally:
        for h in handles:
            h.remove()

    return FiLMResult(table=pd.DataFrame(rows))


# ─── Part F.3 — Attention weight extraction ────────────────────────────

@dataclass
class AttentionMapsResult:
    maps: dict[str, np.ndarray]            # key = (window_label, t) → (L, L) softmaxed
    titles: list[str]


def attention_maps(
    generator,
    val_windows_by_label: dict[str, np.ndarray],   # label → (L, F)
    timesteps: list[int] = (900, 500, 100),
    norm_stats=None,
    device: str = "cuda",
) -> AttentionMapsResult:
    """Re-compute first-block first-head attention by tapping into to_q/to_k.

    Replicates the denoiser's pre-attention forward pass out-of-band so we
    can extract softmax(QKᵀ) at block 0 head 0. Both conditioning paths
    supported (FiLM in v2-v4; AdaLN-Zero in v5+).
    """
    from einops import rearrange
    denoiser = generator.denoiser
    # EDM-wrapped models: unwrap to the inner TradesStyleDenoiser. The EDM
    # preconditioning math doesn't affect the attention pattern itself
    # (preconditioning only rescales x and the noise embedding); we still
    # extract attention from block 0 of the inner denoiser.
    is_edm = hasattr(denoiser, "inner")
    if is_edm:
        denoiser = denoiser.inner
    is_adaln = hasattr(denoiser, "adaln_blocks")

    if is_adaln:
        block = denoiser.adaln_blocks[0]
    else:
        block = denoiser.blocks[0]
    d = block.d_model
    h = block.num_heads

    maps: dict[str, np.ndarray] = {}
    titles: list[str] = []
    cond = torch.tensor([[0, 0, 1, 1]], dtype=torch.long, device=device)

    generator.eval()
    with torch.no_grad():
        for label, x_np in val_windows_by_label.items():
            for t in timesteps:
                x = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(0).to(device)
                t_tensor = torch.tensor([int(t)], dtype=torch.long, device=device)
                ctx = generator.regime_embed(cond)

                if is_adaln:
                    # AdaLN forward: in_proj + pos, then build c = t_emb + regime_proj(ctx),
                    # then block 0's modulated pre-norm before attention.
                    h_in = denoiser.in_proj(x)
                    L = h_in.shape[1]
                    h_in = h_in + denoiser.pos_embed[:L].unsqueeze(0)
                    t_emb = denoiser.t_mlp(denoiser.t_sinusoidal_table[t_tensor])
                    c = t_emb + denoiser.regime_proj(ctx)
                    # Block 0's modulation MLP gives 6 chunks; we need just γ_attn, β_attn
                    # for the pre-attention LN+modulate (the α scales the residual *after*
                    # attention which doesn't affect the attention pattern itself).
                    gamma1, beta1, _alpha1, _g2, _b2, _a2 = block.adaLN_modulation(c).chunk(6, dim=-1)
                    h_in = block.norm1(h_in) * (1.0 + gamma1.unsqueeze(1)) + beta1.unsqueeze(1)
                else:
                    # FiLM forward: in_proj + pos + t_embed_table[t] + film_in + layer_norm,
                    # matching TradesStyleDenoiser._forward_film.
                    h_in = denoiser.in_proj(x)
                    L = h_in.shape[1]
                    h_in = h_in + denoiser.pos_embed[:L].unsqueeze(0)
                    h_in = h_in + denoiser.t_embed_table[t_tensor].unsqueeze(1)
                    h_in = denoiser.film_in(h_in, ctx)
                    h_in = denoiser.layer_norm(h_in)

                # Attention math (block 0, head 0) — same for both paths
                q = block.to_q(h_in); k = block.to_k(h_in)
                q = rearrange(q, "b l (h j) -> b h l j", h=h)
                k = rearrange(k, "b l (h j) -> b h l j", h=h)
                q = q * (d ** -0.5)
                e = torch.einsum("b h l j, b h k j -> b h l k", q, k)
                att = torch.softmax(e, dim=-1)
                map_arr = att[0, 0].cpu().numpy()                # (L, L), head 0
                key = f"{label}_t{t}"
                maps[key] = map_arr
                titles.append(key)

    return AttentionMapsResult(maps=maps, titles=titles)


# ─── Part F.4 — Guidance weight sweep ──────────────────────────────────

@dataclass
class GuidanceSweepResult:
    table: pd.DataFrame                    # cols: w, mean_abs_return, mean_spread, std_return, trade_fraction, mean_bid_sz
    real_targets: dict[str, float]


def guidance_sweep(
    generator,
    schedule,
    norm_stats,
    *,
    regime_label: str = "high_vol",
    regime_to_condition: dict[str, list[int]],
    real_per_day_for_target: list[pl.DataFrame],
    ws: list[float] = (0.0, 0.5, 1.0, 2.0, 3.0, 5.0),
    n_seeds: int = 50,
    seq_len: int = 256,
    device: str = "cuda",
) -> GuidanceSweepResult:
    """Sample 50 windows at each w, decode, compute regime statistics.

    Auto-detects DDIM vs EDM via the schedule type and uses the appropriate
    sampler. Output table is the same shape regardless of sampler.
    """
    from ..generator.decode import decode_window_to_dataframe
    from ..data.dataset import N_FEATURES

    is_edm = hasattr(schedule, "sigma_data") and hasattr(schedule, "get_sigmas")
    if is_edm:
        from ..generator.edm import edm_sample
    else:
        from ..generator.sample import ddim_sample

    # Real targets: aggregate over real-day events partitioned to this regime
    target_stats = {}
    real_concat = pl.concat([d.filter(pl.col("regime_label") == regime_label) for d in real_per_day_for_target])
    if real_concat.height > 0:
        r = real_concat["mid_return"].fill_null(0).to_numpy()
        target_stats = {
            "mean_abs_return": float(np.mean(np.abs(r))),
            "mean_spread": float(real_concat["spread"].mean()),
            "std_return": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
            "trade_fraction": float((real_concat["event_type"] == "trade").mean()),
            "mean_bid_sz": float(real_concat["bid_sz"].mean()),
        }

    cond_vec = regime_to_condition[regime_label]
    rows = []
    for w in ws:
        per_seed_stats = []
        for seed in range(n_seeds):
            cond = torch.tensor([cond_vec], dtype=torch.long, device=device)
            null = generator.regime_embed.null_condition(1, device)
            if is_edm:
                x = edm_sample(
                    predict_x0=generator.predict_noise,
                    shape=(1, seq_len, N_FEATURES),
                    schedule=schedule,
                    condition=cond,
                    null_condition=null,
                    guidance_weight=float(w),
                    n_steps=50,
                    device=device,
                    seed=seed,
                    second_order=True,
                )
            else:
                x = ddim_sample(
                    eps_theta=generator.predict_noise,
                    shape=(1, seq_len, N_FEATURES),
                    schedule=schedule,
                    condition=cond,
                    null_condition=null,
                    guidance_weight=float(w),
                    n_steps=200,
                    device=device,
                    seed=seed,
                    eta=0.0,
                    x0_clip=4.0,
                )
            df = decode_window_to_dataframe(
                x[0],
                condition=cond[0],
                ticker="INTC",
                start_ts_ns=1_700_000_000_000_000_000 + seed * 86_400_000_000_000,
                norm_stats=norm_stats,
            )
            r = df["mid_return"].fill_null(0).to_numpy()
            per_seed_stats.append({
                "mean_abs_return": float(np.mean(np.abs(r))),
                "mean_spread": float(df["spread"].mean()),
                "std_return": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
                "trade_fraction": float((df["event_type"] == "trade").mean()),
                "mean_bid_sz": float(df["bid_sz"].mean()),
            })
        if per_seed_stats:
            df_s = pd.DataFrame(per_seed_stats)
            row = {"w": w}
            for k in df_s.columns:
                row[f"{k}_mean"] = float(df_s[k].mean())
                row[f"{k}_std"] = float(df_s[k].std(ddof=1))
            rows.append(row)
    return GuidanceSweepResult(table=pd.DataFrame(rows), real_targets=target_stats)


# ─── Part F.5 — Per-feature loss ───────────────────────────────────────

@dataclass
class PerFeatureMSEResult:
    feature_names: list[str]
    mse_per_feature: np.ndarray


def per_feature_mse(
    generator,
    schedule,
    val_windows: list[np.ndarray],
    *,
    t: int = 300,
    n_samples: int = 200,
    device: str = "cuda",
    feature_names: list[str] | None = None,
) -> PerFeatureMSEResult:
    """At a fixed noise level, MSE per feature channel averaged over n_samples.

    DDIM/DDPM: noise level is integer timestep t (default 300, mid-range).
    EDM: noise level is σ ≈ exp(P_mean) ≈ 0.30 (the median training σ).
    EDM model is x_0-parameterized so target is x_0 itself.
    """
    rng = np.random.default_rng(0)
    is_edm = hasattr(schedule, "sigma_data") and hasattr(schedule, "get_sigmas")
    if not val_windows:
        n_features = generator.denoiser.n_features
        return PerFeatureMSEResult(
            feature_names=feature_names or [f"f{i}" for i in range(n_features)],
            mse_per_feature=np.zeros(n_features),
        )

    L = val_windows[0].shape[0]
    F = val_windows[0].shape[1]
    stacked = np.stack([w[:L] for w in val_windows], axis=0)
    idx = rng.integers(0, stacked.shape[0], size=min(n_samples, stacked.shape[0]))
    x0 = torch.from_numpy(stacked[idx].astype(np.float32)).to(device)
    B = x0.shape[0]
    cond = torch.tensor([[0, 0, 1, 1]] * B, dtype=torch.long, device=device)
    eps_true = torch.randn_like(x0)

    if is_edm:
        sigma = torch.full((B,), 0.30, device=device)         # median EDM σ
        sigma_view = sigma.view(B, *([1] * (x0.dim() - 1)))
        x_t = x0 + sigma_view * eps_true
    else:
        a_bar = schedule.alphas_cumprod.to(device)
        t_tensor = torch.full((B,), int(t), dtype=torch.long, device=device)
        a_t = a_bar[t]
        x_t = torch.sqrt(a_t) * x0 + torch.sqrt(1 - a_t) * eps_true

    generator.eval()
    with torch.no_grad():
        if is_edm:
            pred = generator.predict_noise(x_t, sigma, cond)   # x_0_pred
            target = x0
        else:
            pred = generator.predict_noise(x_t, t_tensor, cond)
            target = eps_true
    mse = ((pred - target) ** 2).mean(dim=(0, 1)).cpu().numpy()    # (F,)

    return PerFeatureMSEResult(
        feature_names=feature_names or [f"f{i}" for i in range(F)],
        mse_per_feature=mse,
    )
