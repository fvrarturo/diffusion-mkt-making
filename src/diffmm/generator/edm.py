"""EDM-style training and sampling (Karras et al. 2022).

Phase D — final architectural variant for the diffusion model. Reformulates
training and sampling around continuous σ (noise level) instead of discrete
t (timestep), with a network preconditioning that normalizes loss magnitude
across noise levels. Empirically SOTA for diffusion on heavy-tailed data.

Module layout (everything EDM-specific is in this file; the only touch to
existing code is `TradesStyleDenoiser.forward_with_t_emb`, an additive
extension point):

  EDMSchedule       — σ schedule + train-time σ sampler (log-normal)
  EDMDenoiser       — preconditioning wrapper around a TradesStyleDenoiser
  edm_sample        — Heun 2nd-order sampler with optional CFG + stochasticity
  EDMTrainer        — Lightning module with σ-weighted loss
  build_edm_generator — factory: GeneratorModel(EDMDenoiser(inner=TRADES))

Reference: Karras, Aittala, Aila, Laine. "Elucidating the Design Space of
Diffusion-Based Generative Models." NeurIPS 2022.

Math summary:
  Forward:  x(σ) = x_0 + σ · ε,    ε ~ N(0, I)
  Network:  D(x; σ; c) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x; c_noise(σ); c)
            with c_skip = σ_data² / (σ² + σ_data²)
                 c_out  = σ · σ_data / sqrt(σ² + σ_data²)
                 c_in   = 1 / sqrt(σ² + σ_data²)
                 c_noise = (1/4) log σ
  Loss:     λ(σ) · ‖D(x_0 + σε; σ; c) − x_0‖²
            with λ(σ) = (σ² + σ_data²) / (σ · σ_data)²
            σ ~ exp(N(P_mean, P_std)) at training time
  Sampler:  Heun 2nd-order, σ schedule with ρ-power spacing.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

try:
    import pytorch_lightning as pl
    LightningModule = pl.LightningModule
except ImportError:
    class LightningModule(torch.nn.Module):
        def log(self, *a, **k): pass

from .conditioning import RegimeEmbedding
from .model import Denoiser, GeneratorModel
from .train import CFGDropoutCurriculum
from .trades_adapter import TradesStyleDenoiser


# ─── σ schedule + training σ sampler ────────────────────────────────────

@dataclass
class EDMSchedule:
    """Noise schedule for EDM training and sampling.

    Defaults follow Karras 2022 §5 (Tables 1, 5). σ_data is dataset-specific;
    0.5 is a generic default for ~N(0, 1) features. For our normalized LOB
    feature distribution it could be calibrated on training data.
    """
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    sigma_data: float = 0.5
    rho: float = 7.0
    P_mean: float = -1.2
    P_std: float = 1.2

    def get_sigmas(self, n_steps: int, device: torch.device | str) -> torch.Tensor:
        """Karras's ρ-power σ schedule: σ_i = (σ_max^(1/ρ) + (i/(N-1))·(σ_min^(1/ρ) − σ_max^(1/ρ)))^ρ.

        Returns a (n_steps + 1,) tensor with the final entry σ_{N} = 0
        (clean data). Sampling iterates pairs (σ_i, σ_{i+1}) for i in 0..N-1.
        """
        i = torch.arange(n_steps, device=device, dtype=torch.float32)
        ramp = i / max(n_steps - 1, 1)
        a = self.sigma_max ** (1.0 / self.rho)
        b = self.sigma_min ** (1.0 / self.rho)
        sigmas = (a + ramp * (b - a)) ** self.rho
        return torch.cat([sigmas, torch.zeros(1, device=device)])

    def sample_train_sigma(
        self, batch_size: int, device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """log-normal σ sampling for training (Karras §5.1).

        log σ ~ N(P_mean, P_std) ⟹ concentrates training on σ ~ exp(P_mean)
        ≈ 0.3 with the default P_mean=-1.2.
        """
        if generator is None:
            log_sigma = torch.randn(batch_size, device=device) * self.P_std + self.P_mean
        else:
            log_sigma = (
                torch.randn(batch_size, device=device, generator=generator)
                * self.P_std + self.P_mean
            )
        return log_sigma.exp()

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        """λ(σ) = (σ² + σ_data²) / (σ · σ_data)² — gives unit weight on the underlying x_0 MSE."""
        return (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2


# ─── Continuous sinusoidal embedding (used for c_noise) ─────────────────

def sinusoidal_continuous(
    value: torch.Tensor, dim: int, n: float = 10000.0,
) -> torch.Tensor:
    """Sinusoidal positional embedding of a continuous scalar (per batch element).

    Same construction as the standard Transformer positional embedding but
    the "position" is a continuous tensor instead of an integer index.

    Args:
        value: (B,) — continuous values to embed (e.g., log(σ)/4)
        dim:   embedding dimension (must be even)

    Returns: (B, dim)
    """
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal embedding dim must be even, got {dim}")
    device = value.device
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(0, half, device=device, dtype=torch.float32)
        * (math.log(n) / max(half - 1, 1))
    )
    args = value.unsqueeze(-1) * freqs.unsqueeze(0)        # (B, half)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ─── EDM-preconditioned denoiser ────────────────────────────────────────

class EDMDenoiser(Denoiser):
    """Preconditioning wrapper around an inner TradesStyleDenoiser.

    Implements the EDM denoising function (Karras 2022 Eq. 7):

        D(x; σ; c) = c_skip(σ)·x + c_out(σ)·F_θ(c_in(σ)·x; c_noise(σ); c)

    where F_θ is the inner denoiser (FiLM or AdaLN-Zero conditioning, as
    determined by the inner's `conditioning_type`).

    The inner denoiser is invoked via its `forward_with_t_emb` extension
    point — the noise level σ enters through a sinusoidal embedding of
    log(σ)/4 supplied as the time embedding. The inner backbone's
    conditioning architecture is fully preserved.

    Forward signature follows the Denoiser ABC: `forward(x, sigma, ctx)`.
    NOTE: the second arg is interpreted as σ (continuous tensor), not as
    integer timestep t. Sampling/training code must respect this contract.
    """

    def __init__(self, inner: TradesStyleDenoiser, sigma_data: float = 0.5):
        super().__init__()
        self.inner = inner
        self.sigma_data = sigma_data
        self.n_features = inner.n_features
        self.d_model = inner.d_model

    def _c_skip(self, sigma: torch.Tensor) -> torch.Tensor:
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def _c_out(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def _c_in(self, sigma: torch.Tensor) -> torch.Tensor:
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def _c_noise(self, sigma: torch.Tensor) -> torch.Tensor:
        return sigma.log() * 0.25

    def forward(
        self, x: torch.Tensor, sigma: torch.Tensor, ctx: torch.Tensor,
    ) -> torch.Tensor:
        """EDM-preconditioned denoise: returns predicted x_0.

        Args:
            x:     (B, L, F) — noisy input
            sigma: (B,) — per-sample noise level (continuous tensor, NOT int t)
            ctx:   (B, embed_dim) — regime context vector
        """
        sigma_b = sigma.view(-1, *([1] * (x.dim() - 1)))   # broadcast over (L, F)
        c_skip = self._c_skip(sigma_b)
        c_out = self._c_out(sigma_b)
        c_in = self._c_in(sigma_b)
        c_noise = self._c_noise(sigma)                     # (B,)

        # Build noise embedding via sinusoidal of c_noise. The inner denoiser
        # treats this exactly like its t-embedding from the integer-t path —
        # FiLM adds it directly to hidden state; AdaLN routes it through t_mlp.
        noise_emb = sinusoidal_continuous(c_noise, self.d_model)
        f_out = self.inner.forward_with_t_emb(c_in * x, noise_emb, ctx)
        return c_skip * x + c_out * f_out


# ─── Heun 2nd-order sampler with optional CFG + stochasticity ───────────

@torch.no_grad()
def edm_sample(
    predict_x0: Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor],
    shape: tuple[int, ...],
    schedule: EDMSchedule,
    *,
    condition: torch.Tensor,
    null_condition: torch.Tensor,
    guidance_weight: float = 1.0,
    n_steps: int = 32,
    device: torch.device | str = "cpu",
    seed: int | None = None,
    s_churn: float = 0.0,
    s_tmin: float = 0.0,
    s_tmax: float = float("inf"),
    s_noise: float = 1.0,
    second_order: bool = True,
    x0_clip: float | None = None,
) -> torch.Tensor:
    """Karras 2nd-order Heun sampler (Algorithm 1/2 in the EDM paper).

    Args:
        predict_x0: callable (x, sigma, c_idx) → x_0_pred. Typically
            `generator.predict_noise` for a GeneratorModel(EDMDenoiser(...)).
            Despite the historical name, it returns x_0 in EDM mode.
        shape: output tensor shape (with batch dim).
        schedule: σ schedule.
        condition: (B, n_axes) long — regime condition.
        null_condition: (B, n_axes) long — null tokens for CFG.
        guidance_weight: w in CFG; D̃ = (1+w)·D(x; σ; c) − w·D(x; σ; ∅).
            CFG on x_0 predictions is algebraically equivalent to CFG on ε
            since both are linear functions of (x_0, ε) at fixed σ.
        n_steps: number of Heun steps. EDM typically needs far fewer than DDIM
            (32-50 vs 100-200) to reach equivalent quality.
        s_churn, s_tmin, s_tmax, s_noise: optional stochasticity (Algorithm 2).
            Default s_churn=0 → fully deterministic Heun.
        second_order: if False, falls back to Euler (faster, lower quality).
        x0_clip: optional bound on |x_0_pred|. EDM doesn't normally need it
            because the preconditioning keeps outputs bounded; provided for
            API consistency with the DDIM path.

    Returns: (B, ...) tensor of generated samples.
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(shape, generator=gen, device=device) * schedule.sigma_max
    else:
        x = torch.randn(shape, device=device) * schedule.sigma_max

    sigmas = schedule.get_sigmas(n_steps, device)
    s_churn_per_step = min(s_churn / n_steps, math.sqrt(2) - 1)

    def _predict_with_cfg(x_in: torch.Tensor, sigma_b: torch.Tensor) -> torch.Tensor:
        x0_cond = predict_x0(x_in, sigma_b, condition)
        if guidance_weight == 0.0:
            return x0_cond
        x0_uncond = predict_x0(x_in, sigma_b, null_condition)
        return (1.0 + guidance_weight) * x0_cond - guidance_weight * x0_uncond

    for i in range(n_steps):
        sigma_i = sigmas[i]
        sigma_next = sigmas[i + 1]

        # Optional stochastic injection (Karras Algorithm 2 §5.2).
        if s_churn > 0 and (s_tmin <= sigma_i <= s_tmax):
            gamma = s_churn_per_step
            sigma_hat = sigma_i * (gamma + 1.0)
            extra_noise = (sigma_hat ** 2 - sigma_i ** 2).clamp(min=0).sqrt()
            x = x + s_noise * torch.randn_like(x) * extra_noise
        else:
            sigma_hat = sigma_i

        # Predict x_0 at (x, σ_hat) with CFG.
        sigma_b = sigma_hat.expand(shape[0])
        x0_pred = _predict_with_cfg(x, sigma_b)
        if x0_clip is not None:
            x0_pred = x0_pred.clamp(-x0_clip, x0_clip)

        # Euler step toward σ_next.
        d_i = (x - x0_pred) / sigma_hat
        dt = sigma_next - sigma_hat
        x_next = x + d_i * dt

        # Heun 2nd-order correction (skip if final step or sigma_next == 0).
        if second_order and float(sigma_next) > 0:
            sigma_b_next = sigma_next.expand(shape[0])
            x0_pred_next = _predict_with_cfg(x_next, sigma_b_next)
            if x0_clip is not None:
                x0_pred_next = x0_pred_next.clamp(-x0_clip, x0_clip)
            d_i_prime = (x_next - x0_pred_next) / sigma_next
            x = x + 0.5 * (d_i + d_i_prime) * dt
        else:
            x = x_next

    return x


# ─── EDM trainer ────────────────────────────────────────────────────────

class EDMTrainer(LightningModule):
    """Lightning module for EDM training.

    Loss: λ(σ) · ‖D(x_0 + σε; σ; c) − x_0‖²
          σ ~ exp(N(P_mean, P_std))

    Reuses Phase B's `CFGDropoutCurriculum` for the conditioning curriculum
    — the curriculum logic is parameterization-independent and applies to
    both DDPM and EDM training.
    """

    def __init__(
        self,
        generator: GeneratorModel,
        schedule: EDMSchedule,
        learning_rate: float = 1e-4,
        cfg_dropout: float = 0.1,
        weight_decay: float = 0.0,
        cfg_dropout_curriculum: CFGDropoutCurriculum | None = None,
    ):
        super().__init__()
        self.generator = generator
        self.lr = learning_rate
        self.cfg_dropout = cfg_dropout
        self.weight_decay = weight_decay
        self.schedule = schedule
        self.curriculum = cfg_dropout_curriculum

    def _current_cfg_dropout(self) -> float:
        if self.curriculum is None or not self.curriculum.enabled:
            return self.cfg_dropout
        epoch = int(getattr(self, "current_epoch", 0))
        return self.curriculum.value_at(epoch)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def _step(self, batch, stage: str) -> torch.Tensor:
        x0, c = batch
        B = x0.shape[0]
        device = x0.device

        # CFG dropout — same protocol as DDPMTrainer.
        active_dropout = self._current_cfg_dropout() if stage == "train" else 0.0
        if active_dropout > 0:
            drop = torch.rand(B, device=device) < active_dropout
            if drop.any():
                c_null = self.generator.regime_embed.null_condition(B, device)
                c = torch.where(drop.unsqueeze(-1), c_null, c)

        # Sample σ from log-normal.
        sigma = self.schedule.sample_train_sigma(B, device)
        sigma_b = sigma.view(-1, *([1] * (x0.dim() - 1)))

        # Add noise: x_t = x_0 + σ · ε
        eps = torch.randn_like(x0)
        x_noisy = x0 + sigma_b * eps

        # Predict x_0 via the EDM-preconditioned denoiser. The Generator
        # model's `predict_noise` callable forwards through the EDMDenoiser
        # and returns x_0_pred (despite the historical method name).
        x0_pred = self.generator.predict_noise(x_noisy, sigma, c)

        # σ-weighted loss.
        per_sample_mse = ((x0_pred - x0) ** 2).mean(
            dim=tuple(range(1, x0.dim()))
        )
        weight = self.schedule.loss_weight(sigma)
        loss = (per_sample_mse * weight).mean()

        trainer = getattr(self, "_trainer", None)
        if trainer is not None:
            self.log(
                f"{stage}/loss", loss, prog_bar=True,
                on_step=(stage == "train"), on_epoch=True,
            )
            if stage == "train":
                self.log("train/cfg_dropout", active_dropout, on_step=False, on_epoch=True)
                self.log("train/sigma_mean", float(sigma.mean()), on_step=False, on_epoch=True)
            step = trainer.global_step
            if stage == "train" and step % 20 == 0:
                print(
                    f"[{stage}] step {step:>6d}  loss {loss.item():.4f}  "
                    f"σ̄={sigma.mean().item():.3f}  pred=edm  p_drop={active_dropout:.2f}",
                    flush=True,
                )
        return loss

    def configure_optimizers(self):
        params = [p for p in self.generator.parameters() if p.requires_grad]
        return torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)


# ─── Factory ────────────────────────────────────────────────────────────

def build_edm_generator(
    *,
    n_features: int,
    d_model: int = 256,
    num_heads: int = 8,
    depth: int = 8,
    max_seq_len: int = 1024,
    embed_dim: int = 128,
    dropout: float = 0.1,
    n_categories_per_axis: tuple[int, ...] = (3, 3, 3, 3),
    conditioning_type: str = "adaln_zero",
    sigma_data: float = 0.5,
) -> GeneratorModel:
    """Build a GeneratorModel(EDMDenoiser(TradesStyleDenoiser(...)), RegimeEmbedding(...)).

    The inner backbone uses `conditioning_type` ∈ {"film", "adaln_zero"}.
    Default is `adaln_zero` — Phase C established that AdaLN-Zero gives
    materially better conditioning differentiation than FiLM, and EDM's
    preconditioning math is independent of the conditioning primitive.

    `sigma_data` should be ≈ the std of normalized training data. Default
    0.5 follows Karras 2022 for ImageNet-scale data; for our normalized LOB
    features (≈ N(0,1) by construction) σ_data=0.5 is conservative.
    """
    regime_embed = RegimeEmbedding(
        n_categories_per_axis=n_categories_per_axis, embed_dim=embed_dim,
    )
    inner = TradesStyleDenoiser(
        n_features=n_features, d_model=d_model, num_heads=num_heads,
        depth=depth, max_seq_len=max_seq_len, ctx_dim=embed_dim,
        dropout=dropout, num_diffusionsteps=1000,    # unused by EDM but required by ctor
        conditioning_type=conditioning_type,
    )
    edm_denoiser = EDMDenoiser(inner=inner, sigma_data=sigma_data)
    return GeneratorModel(denoiser=edm_denoiser, regime_embed=regime_embed)
