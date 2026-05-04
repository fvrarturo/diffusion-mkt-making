"""Generator wrapper around the underlying denoiser.

The Denoiser ABC is the contract that both the TRADES adapter and the
MinimalDenoiser (test-only) implement. The training loop and the sampler
talk to GeneratorModel, which in turn calls the denoiser.
"""
from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

import torch
import torch.nn as nn

from .conditioning import FiLMLayer, RegimeEmbedding


class Denoiser(nn.Module):
    """Predicts the noise ε given (x_t, t, context). Subclass to implement."""

    @abstractmethod
    def forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        ...


class GeneratorModel(nn.Module):
    """Pairs a Denoiser with a regime-embedding module.

    The denoiser sees `ctx`, a single (B, embed_dim) context vector. It is
    free to consume that context however it wants — typically through FiLM
    layers inside its blocks, fed by `ctx`.
    """
    def __init__(self, denoiser: Denoiser, regime_embed: RegimeEmbedding):
        super().__init__()
        self.denoiser = denoiser
        self.regime_embed = regime_embed

    def predict_noise(self, x_t: torch.Tensor, t: torch.Tensor, c_idx: torch.Tensor) -> torch.Tensor:
        ctx = self.regime_embed(c_idx)
        return self.denoiser(x_t, t, ctx)

    @classmethod
    def from_pretrained(cls, checkpoint_path: str | Path, regime_embed: RegimeEmbedding) -> GeneratorModel:
        """Filled in by trades_adapter.load_pretrained_model() once we inspect DeepMarket."""
        raise NotImplementedError("see trades_adapter.py — implemented after DeepMarket inspection")


# ─── Test-only minimal denoiser ─────────────────────────────────────────
# Used by tests/test_generator_smoke.py to exercise training/sampling
# without needing TRADES. NOT used in production.

def _sinusoidal_time_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    """Sinusoidal positional embedding of timestep, like Transformer pos enc."""
    half = dim // 2
    freqs = torch.exp(
        -torch.arange(half, device=t.device, dtype=torch.float32)
        * (torch.log(torch.tensor(10000.0)) / max(half - 1, 1))
    )
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2 == 1:
        emb = nn.functional.pad(emb, (0, 1))
    return emb


class MinimalDenoiser(Denoiser):
    """Tiny per-token MLP denoiser with FiLM conditioning. For tests only."""
    def __init__(self, n_features: int, hidden_dim: int = 64, ctx_dim: int = 128, time_dim: int = 64):
        super().__init__()
        self.in_proj = nn.Linear(n_features, hidden_dim)
        self.t_proj = nn.Linear(time_dim, hidden_dim)
        self.time_dim = time_dim
        self.ctx_dim = ctx_dim
        self.film = FiLMLayer(context_dim=ctx_dim, hidden_dim=hidden_dim)
        self.mid = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.out = nn.Linear(hidden_dim, n_features)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        # x_t: (B, L, F), t: (B,), ctx: (B, ctx_dim)
        h = self.in_proj(x_t)
        t_emb = _sinusoidal_time_embedding(t, self.time_dim)
        h = h + self.t_proj(t_emb).unsqueeze(1)
        h = self.film(h, ctx)
        h = self.mid(h)
        return self.out(h)
