"""DDIM sampling with classifier-free guidance.

References:
  Ho, Jain & Abbeel (2020) — DDPM
  Song, Meng & Ermon (2021) — DDIM
  Ho & Salimans (2022) — Classifier-Free Diffusion Guidance
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor


@dataclass
class DDIMSchedule:
    betas: Tensor
    # "eps" — model predicts the noise ε_θ(x_t, t, c) (Ho et al. 2020 standard).
    # "v"   — model predicts v = √ᾱ_t · ε − √(1-ᾱ_t) · x_0 (Salimans & Ho 2022).
    # The two are mathematically related but train under different loss
    # landscapes. v-prediction has more uniform loss-vs-SNR profile and
    # generally recovers heavy tails better on densities with rare extreme
    # events — the documented motivation for switching after Work6_update §5
    # showed ε-prediction undertrains kurtosis on INTC event-level returns.
    prediction_type: str = "eps"

    @property
    def alphas(self) -> Tensor:
        return 1.0 - self.betas

    @property
    def alphas_cumprod(self) -> Tensor:
        return torch.cumprod(self.alphas, dim=0)

    @classmethod
    def linear(
        cls, T: int = 1000, beta_start: float = 1e-4, beta_end: float = 2e-2,
        prediction_type: str = "eps",
    ) -> DDIMSchedule:
        return cls(betas=torch.linspace(beta_start, beta_end, T), prediction_type=prediction_type)

    @classmethod
    def cosine(cls, T: int = 1000, s: float = 0.008, prediction_type: str = "eps") -> DDIMSchedule:
        # Nichol & Dhariwal (2021) cosine schedule.
        steps = T + 1
        t = torch.linspace(0, T, steps) / T
        f = torch.cos(((t + s) / (1 + s)) * torch.pi * 0.5) ** 2
        alphas_cumprod = f / f[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return cls(betas=torch.clamp(betas, max=0.999), prediction_type=prediction_type)


@torch.no_grad()
def ddim_sample(
    eps_theta: Callable[[Tensor, Tensor, Tensor], Tensor],
    shape: tuple[int, ...],
    schedule: DDIMSchedule,
    *,
    condition: Tensor,
    null_condition: Tensor,
    guidance_weight: float = 3.0,
    n_steps: int = 20,
    device: torch.device | str = "cpu",
    seed: int | None = None,
    eta: float = 0.0,
    x0_clip: float | None = 4.0,
) -> Tensor:
    """Reverse-DDIM sampling with classifier-free guidance.

    Args:
        eps_theta: model forward pass with signature (x_t, t_int, c) -> pred.
            The argument is named `eps_theta` for historical reasons; in
            `schedule.prediction_type == "v"` mode the same callable is
            interpreted as a v-predictor instead of an ε-predictor. CFG
            combination still operates on the raw model output (algebraically
            valid for both ε and v since v is a linear function of x_0 and ε).
        shape: output tensor shape, including batch dim.
        schedule: precomputed beta schedule. Its `prediction_type` selects
            how to convert the model output into x_0_pred and ε_pred.
        condition: (B, n_axes) long — the regime condition.
        null_condition: (B, n_axes) long — null tokens for the unconditional branch.
        guidance_weight: w in CFG. For ε-prediction: ε̃ = (1+w)·ε(c) − w·ε(∅).
            For v-prediction the same combination is applied to v
            (Salimans & Ho 2022 §4.2 — CFG works identically on v).
            w=0 → unconditional.
        n_steps: number of DDIM steps S, S ≤ T.
        eta: 0 → deterministic (DDIM); 1 → DDPM-equivalent stochasticity.
        x0_clip: if not None, clamp x0_pred to [-x0_clip, +x0_clip] after each
            step. Prevents catastrophic blow-up at high-noise timesteps where
            sqrt(alpha) → 0 magnifies imperfect predictions (Ho et al. 2020 §4;
            Saharia et al. / Imagen 2022 dynamic thresholding). Data is
            normalized to ~N(0, 1) so ±4 σ is generous.

    Returns: (B, ...) tensor x_0.
    """
    if seed is not None:
        gen = torch.Generator(device=device).manual_seed(seed)
        x = torch.randn(shape, generator=gen, device=device)
    else:
        x = torch.randn(shape, device=device)

    a_bar = schedule.alphas_cumprod.to(device)
    T = a_bar.shape[0]

    # Subsampled timesteps: T-1, ..., 0.
    timesteps = torch.linspace(T - 1, 0, n_steps + 1, dtype=torch.long, device=device)

    pred_type = schedule.prediction_type
    if pred_type not in ("eps", "v"):
        raise ValueError(f"unknown prediction_type {pred_type!r}; expected 'eps' or 'v'")

    for i in range(n_steps):
        t = timesteps[i]
        t_next = timesteps[i + 1]
        a_t = a_bar[t]
        a_next = a_bar[t_next] if t_next >= 0 else torch.tensor(1.0, device=device)

        t_batch = t.expand(shape[0])
        out_cond = eps_theta(x, t_batch, condition)
        if guidance_weight != 0.0:
            out_uncond = eps_theta(x, t_batch, null_condition)
            out = (1.0 + guidance_weight) * out_cond - guidance_weight * out_uncond
        else:
            out = out_cond

        # Convert raw model output → (x0_pred, eps_pred) based on prediction_type.
        if pred_type == "eps":
            eps = out
            x0_pred = (x - torch.sqrt(1.0 - a_t) * eps) / torch.sqrt(a_t)
        else:  # "v"
            # x_0 = √ᾱ_t · x_t − √(1−ᾱ_t) · v;  ε = √(1−ᾱ_t) · x_t + √ᾱ_t · v
            v = out
            x0_pred = torch.sqrt(a_t) * x - torch.sqrt(1.0 - a_t) * v
            eps = torch.sqrt(1.0 - a_t) * x + torch.sqrt(a_t) * v

        if x0_clip is not None:
            x0_pred = x0_pred.clamp(-x0_clip, x0_clip)
            # When v-prediction's x0_pred is clipped, recompute eps to stay
            # consistent with the clipped x0 (otherwise the DDIM step uses
            # an eps that no longer matches the clipped x0_pred).
            if pred_type == "v":
                eps = (x - torch.sqrt(a_t) * x0_pred) / torch.sqrt(torch.clamp(1.0 - a_t, min=1e-12))

        if eta > 0:
            sigma = eta * torch.sqrt((1 - a_next) / (1 - a_t)) * torch.sqrt(1 - a_t / a_next)
            noise = sigma * torch.randn_like(x)
        else:
            sigma = torch.tensor(0.0, device=device)
            noise = 0.0

        dir_xt = torch.sqrt(torch.clamp(1 - a_next - sigma ** 2, min=0.0)) * eps
        x = torch.sqrt(a_next) * x0_pred + dir_xt + noise

    return x
