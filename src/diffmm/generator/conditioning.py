"""FiLM conditioning for the diffusion denoiser.

Embeds the regime vector c = (c_vol, c_vpin, c_imb, c_tod), each ∈ {0,1,2},
into a single context vector and turns it into per-layer modulation
parameters γ, β. Includes a "null" token per axis used by classifier-free
guidance dropout.

References:
  Perez et al. (2018) — FiLM
  Ho & Salimans (2022) — Classifier-Free Diffusion Guidance
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RegimeEmbedding(nn.Module):
    """Embed (c_vol, c_vpin, c_imb, c_tod) into a single vector.

    Each axis has its own embedding table of size (n_cats + 1, embed_dim).
    The (n_cats)-th index per axis is the "null" token used for CFG dropout
    and for unconditional sampling.
    """
    def __init__(
        self,
        n_categories_per_axis: tuple[int, ...] = (3, 3, 3, 3),
        embed_dim: int = 128,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_categories_per_axis = tuple(n_categories_per_axis)
        self.per_axis = nn.ModuleList([
            nn.Embedding(n + 1, embed_dim) for n in n_categories_per_axis
        ])
        # null index per axis = n (one past the last real category)
        self.register_buffer(
            "null_idx",
            torch.tensor(n_categories_per_axis, dtype=torch.long),
        )

    def null_condition(self, batch_size: int, device: torch.device | str) -> torch.Tensor:
        return self.null_idx.to(device).unsqueeze(0).expand(batch_size, -1).contiguous()

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        """c: (B, n_axes) long → (B, embed_dim)."""
        out = self.per_axis[0](c[:, 0])
        for i in range(1, len(self.per_axis)):
            out = out + self.per_axis[i](c[:, i])
        return out


class FiLMLayer(nn.Module):
    """h ↦ (1 + γ) ⊙ h + β, with (γ, β) predicted from a context vector.

    Weight init: small Gaussian (σ=0.02), bias zero. Earlier versions of this
    code used pure-zero init for "identity at step 0" stability during
    fine-tuning, but Work5 §F.2 (and the F.1b cosine diagnostic) showed that
    starting from identity caused FiLM to never specialize across regimes —
    γ and β stayed near identical for every condition after 5 epochs of
    fine-tuning. Random init breaks the symmetry: from step 0, FiLM has
    nontrivial gradients with respect to ctx, and the model is forced to
    decide whether to keep or attenuate the conditioning signal rather than
    starting in a position where the easy answer is "ignore it".

    σ=0.02 is small enough that the initial perturbation is well within the
    range of bf16 precision and does not destabilize fine-tuning of the
    pretrained backbone.
    """
    def __init__(self, context_dim: int, hidden_dim: int):
        super().__init__()
        self.proj = nn.Linear(context_dim, 2 * hidden_dim)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, h: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        gb = self.proj(ctx)
        gamma, beta = gb.chunk(2, dim=-1)
        if h.dim() == 3:                # (B, L, D) — broadcast over L
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return (1.0 + gamma) * h + beta
