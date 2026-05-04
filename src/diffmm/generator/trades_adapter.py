"""TRADES-style denoiser for the canonical event-tape schema.

We adopt the TRADES architecture (transformer encoder + sinusoidal positional
and diffusion-timestep embeddings, Berti et al. 2025) but train from scratch
because the published TRADES checkpoint expects LOBSTER message-level data
(6-feature orders + 40-feature LOB snapshot per step) which we do not have
in our TAQ NBBO setup. The pretrained TRADES weights can still be used as an
ablation baseline once we have LOBSTER data; until then this module IS the
production denoiser.

Differences from TRADES:
  - Input/output: our 9-feature canonical schema (see data.dataset.FEATURE_COLUMNS).
  - Conditioning: switchable between **FiLM** (Phase B / v4 production) and
    **AdaLN-Zero** (Phase C / v5+, Peebles & Xie 2023 DiT). The AdaLN-Zero
    path replaces the FiLM-injected regime+timestep with per-block adaptive
    layer norm + zero-initialized residual gate, which gives ~13× more
    conditioning capacity per layer and starts as exact identity (every
    block is a no-op until the modulation MLP learns nontrivial values).
    Set `conditioning_type="film"` for backwards compat with v2/v3/v3.5/v4
    checkpoints, `conditioning_type="adaln_zero"` for v5+.
  - Output: predicts ε or v depending on schedule (see sample.py / train.py).

Encode/decode helpers convert between canonical Parquet windows and the
denoiser's tensor format. The denoiser is the contract — encode/decode just
slice/concat numpy.
"""
from __future__ import annotations

import math
from pathlib import Path

import polars as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..data.dataset import FEATURE_COLUMNS, NormStats
from .conditioning import FiLMLayer, RegimeEmbedding
from .model import Denoiser, GeneratorModel


# ─── building blocks (lifted from TRADES with minor cleanup) ────────────

def sinusoidal_positional_embedding(seq_len: int, dim: int, n: float = 10000.0) -> torch.Tensor:
    if dim % 2 != 0:
        raise ValueError(f"sinusoidal embedding dim must be even, got {dim}")
    pos = torch.arange(0, seq_len).unsqueeze(1).float()
    denom = torch.pow(n, 2 * torch.arange(0, dim // 2).float() / dim)
    emb = torch.zeros(seq_len, dim)
    emb[:, 0::2] = torch.sin(pos / denom)
    emb[:, 1::2] = torch.cos(pos / denom)
    return emb


class TransformerBlockSelfAtt(nn.Module):
    """Self-attention block with PreLN MLP. Matches TRADES' TransformerBlockSelfAtt.

    Used by the FiLM-conditioning path (v2/v3/v3.5/v4). The AdaLN path uses
    AdaLNTransformerBlock instead.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.PReLU(init=0.01),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x); k = self.to_k(x); v = self.to_v(x)
        q, k, v = (rearrange(t, "b l (h j) -> b h l j", h=self.num_heads) for t in (q, k, v))
        q = q * (self.d_model ** -0.5)
        e = torch.einsum("b h l j, b h k j -> b h l k", q, k)
        att = torch.nan_to_num(F.softmax(e, dim=-1))
        out_att = torch.einsum("b h l k, b h k j -> b h l j", att, v)
        out_att = rearrange(out_att, "b h l j -> b l (h j)")
        out_att = self.to_out(out_att)
        out_att = self.layer_norm1(out_att + x)
        out = self.mlp(out_att)
        return self.layer_norm2(out + out_att)


# ─── AdaLN-Zero blocks (DiT-style, Peebles & Xie 2023) ──────────────────

def _modulate(h: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    """h * (1 + γ) + β with γ, β broadcast over the sequence axis."""
    return h * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class AdaLNTransformerBlock(nn.Module):
    """Self-attention + MLP block with AdaLN-Zero conditioning.

    Each block has 6 modulation parameters per forward pass, all derived from a
    single conditioning vector c via one linear projection (zero-initialized):

        γ_attn, β_attn, α_attn, γ_mlp, β_mlp, α_mlp = chunk(Linear(SiLU(c)))

    The forward pass is (DiT §3.1):

        h = h + α_attn · Attn( LN(h) · (1+γ_attn) + β_attn )
        h = h + α_mlp  · MLP ( LN(h) · (1+γ_mlp ) + β_mlp  )

    LN here is `LayerNorm(elementwise_affine=False)` — the γ/β provided by
    AdaLN replace the LN's learnable scale/shift.

    Zero-init of the modulation projection means at step 0:
        γ_attn = β_attn = α_attn = γ_mlp = β_mlp = α_mlp = 0
    so the block is exactly the identity (h passes through unchanged). The
    optimizer learns conditioning from a clean slate, with the residual gate
    acting as a learnable strength dial — addresses the FiLM-collapse failure
    mode documented in Work5 §F.2 (where Phase B's random-init FiLM only
    pushed γ-dev std/mean to 2.88%, far short of the 5% target).
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ctx_dim: int,
        dropout: float = 0.1,
        mlp_ratio: int = 4,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.d_model = d_model
        self.num_heads = num_heads

        # Pre-norms with NO learnable affine (AdaLN provides scale + shift).
        self.norm1 = nn.LayerNorm(d_model, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(d_model, elementwise_affine=False)

        # Self-attention projections (no bias, matching DiT and TRADES).
        self.to_q = nn.Linear(d_model, d_model, bias=False)
        self.to_k = nn.Linear(d_model, d_model, bias=False)
        self.to_v = nn.Linear(d_model, d_model, bias=False)
        self.to_out = nn.Linear(d_model, d_model, bias=False)

        # MLP — DiT uses GELU; we keep PReLU to match TRADES' choice for
        # consistency with the FiLM path. Both are well-studied here.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_ratio * d_model),
            nn.PReLU(init=0.01),
            nn.Linear(mlp_ratio * d_model, d_model),
            nn.Dropout(dropout),
        )

        # AdaLN modulation MLP: c → SiLU → Linear → 6 × d_model.
        # Zero-init the Linear so the block is identity at step 0.
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(ctx_dim, 6 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def _self_attention(self, x: torch.Tensor) -> torch.Tensor:
        q = self.to_q(x); k = self.to_k(x); v = self.to_v(x)
        q, k, v = (rearrange(t, "b l (h j) -> b h l j", h=self.num_heads) for t in (q, k, v))
        q = q * (self.d_model ** -0.5)
        e = torch.einsum("b h l j, b h k j -> b h l k", q, k)
        att = torch.nan_to_num(F.softmax(e, dim=-1))
        out_att = torch.einsum("b h l k, b h k j -> b h l j", att, v)
        out_att = rearrange(out_att, "b h l j -> b l (h j)")
        return self.to_out(out_att)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        # c : (B, ctx_dim) → 6 × (B, d_model)
        gamma1, beta1, alpha1, gamma2, beta2, alpha2 = self.adaLN_modulation(c).chunk(6, dim=-1)
        # Attention sub-block with modulated pre-norm + gated residual
        h_norm = _modulate(self.norm1(h), gamma1, beta1)
        h = h + alpha1.unsqueeze(1) * self._self_attention(h_norm)
        # MLP sub-block with modulated pre-norm + gated residual
        h_norm = _modulate(self.norm2(h), gamma2, beta2)
        h = h + alpha2.unsqueeze(1) * self.mlp(h_norm)
        return h


class AdaLNFinalLayer(nn.Module):
    """Final modulated LayerNorm before output projection.

    Same zero-init trick: at start, γ=β=0 so the layer is just LN(h).
    """

    def __init__(self, d_model: int, ctx_dim: int):
        super().__init__()
        self.norm_final = nn.LayerNorm(d_model, elementwise_affine=False)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(ctx_dim, 2 * d_model, bias=True),
        )
        nn.init.zeros_(self.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.adaLN_modulation[-1].bias)

    def forward(self, h: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.adaLN_modulation(c).chunk(2, dim=-1)
        return _modulate(self.norm_final(h), gamma, beta)


# ─── the production denoiser ────────────────────────────────────────────

class TradesStyleDenoiser(Denoiser):
    """Predicts ε(x_t, t, regime) — or v(x_t, t, regime) under v-pred — as a
    transformer over an event window.

    Tensor flow (FiLM path, v2-v4):
        x_t : (B, L, F_in)              feature_dim = len(FEATURE_COLUMNS) = 9
        t   : (B,) long                 diffusion timestep
        ctx : (B, embed_dim)            regime embedding from RegimeEmbedding

      ↓ in_proj: F_in → d_model
        + sinusoidal(L, d_model)        positional
        + sinusoidal(t, d_model)        diffusion-timestep, broadcast over L
        + FiLM(ctx)                     regime modulation (start)
      ↓ N × TransformerBlockSelfAtt
      ↓ FiLM(ctx)                       regime modulation (end)
      ↓ out_proj: d_model → F_in        prediction
        out : (B, L, F_in)

    Tensor flow (AdaLN-Zero path, v5+):
        x_t : (B, L, F_in)
        t   : (B,) long
        ctx : (B, embed_dim)

      ↓ in_proj: F_in → d_model
        + sinusoidal(L, d_model)        positional
      ↓ Build conditioning vector
        c = t_mlp(sinusoidal(t)) + regime_proj(ctx)     in (B, d_model)
      ↓ N × AdaLNTransformerBlock(h, c)
      ↓ AdaLNFinalLayer(h, c)
      ↓ out_proj: d_model → F_in        prediction
        out : (B, L, F_in)

    The diffusion timestep is no longer added directly to the hidden state
    in the AdaLN path — it goes through the modulation MLP alongside the
    regime, which is the DiT design and avoids time-embedding "leakage" into
    the activation representation.
    """

    def __init__(
        self,
        n_features: int,
        d_model: int = 256,
        num_heads: int = 8,
        depth: int = 8,
        max_seq_len: int = 1024,
        ctx_dim: int = 128,
        dropout: float = 0.1,
        num_diffusionsteps: int = 1000,
        conditioning_type: str = "film",
    ):
        super().__init__()
        if conditioning_type not in ("film", "adaln_zero"):
            raise ValueError(
                f"conditioning_type must be 'film' or 'adaln_zero', got {conditioning_type!r}"
            )
        self.n_features = n_features
        self.d_model = d_model
        self.depth = depth
        self.conditioning_type = conditioning_type

        self.in_proj = nn.Linear(n_features, d_model)
        self.out_proj = nn.Linear(d_model, n_features)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        # Precomputed sinusoidal tables. Registered as buffers so they move with .to(device).
        self.register_buffer(
            "pos_embed",
            sinusoidal_positional_embedding(max_seq_len, d_model),
            persistent=False,
        )

        if conditioning_type == "film":
            # FiLM path — direct timestep additive into hidden state, plus FiLM modulation.
            self.register_buffer(
                "t_embed_table",
                sinusoidal_positional_embedding(num_diffusionsteps, d_model),
                persistent=False,
            )
            self.film_in = FiLMLayer(context_dim=ctx_dim, hidden_dim=d_model)
            self.blocks = nn.ModuleList([
                TransformerBlockSelfAtt(d_model=d_model, num_heads=num_heads, dropout=dropout)
                for _ in range(depth)
            ])
            self.film_out = FiLMLayer(context_dim=ctx_dim, hidden_dim=d_model)
            self.layer_norm = nn.LayerNorm(d_model)
        else:
            # AdaLN-Zero path — conditioning bundled into c = t_emb + ctx_proj, all routed
            # through per-block modulation MLPs (zero-init for identity-at-start).
            self.register_buffer(
                "t_sinusoidal_table",
                sinusoidal_positional_embedding(num_diffusionsteps, d_model),
                persistent=False,
            )
            # Project sinusoidal time embedding through MLP — DiT pattern.
            self.t_mlp = nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.SiLU(),
                nn.Linear(d_model, d_model),
            )
            # Project regime embedding into d_model space (so c is in d_model).
            self.regime_proj = nn.Linear(ctx_dim, d_model)
            self.adaln_blocks = nn.ModuleList([
                AdaLNTransformerBlock(
                    d_model=d_model, num_heads=num_heads, ctx_dim=d_model, dropout=dropout,
                )
                for _ in range(depth)
            ])
            self.adaln_final = AdaLNFinalLayer(d_model=d_model, ctx_dim=d_model)

    def _forward_film(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, L, _ = x_t.shape
        h = self.in_proj(x_t)                                  # (B, L, d)
        h = h + self.pos_embed[:L].unsqueeze(0)                # add positional
        h = h + self.t_embed_table[t].unsqueeze(1)             # add diffusion-time
        h = self.film_in(h, ctx)                               # regime mod (start)
        h = self.layer_norm(h)
        for block in self.blocks:
            h = block(h)
        h = self.film_out(h, ctx)                              # regime mod (end)
        return self.out_proj(h)

    def _forward_adaln(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        B, L, _ = x_t.shape
        h = self.in_proj(x_t) + self.pos_embed[:L].unsqueeze(0)            # (B, L, d_model)
        # Build conditioning vector c = t_emb + regime_proj(ctx) in d_model space.
        t_emb = self.t_mlp(self.t_sinusoidal_table[t])                     # (B, d_model)
        c = t_emb + self.regime_proj(ctx)                                  # (B, d_model)
        for block in self.adaln_blocks:
            h = block(h, c)
        h = self.adaln_final(h, c)
        return self.out_proj(h)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        if self.conditioning_type == "film":
            return self._forward_film(x_t, t, ctx)
        return self._forward_adaln(x_t, t, ctx)


# ─── factory + canonical-tensor I/O ─────────────────────────────────────

def build_generator(
    *,
    n_features: int,
    d_model: int = 256,
    num_heads: int = 8,
    depth: int = 8,
    max_seq_len: int = 1024,
    embed_dim: int = 128,
    dropout: float = 0.1,
    num_diffusionsteps: int = 1000,
    n_categories_per_axis: tuple[int, ...] = (3, 3, 3, 3),
    conditioning_type: str = "film",
) -> GeneratorModel:
    """Build a fresh TRADES-style GeneratorModel ready for training from scratch.

    `conditioning_type`:
      - `"film"` (default for backwards compat with v2-v4 checkpoints): pre-norm
        transformer + FiLM modulation at start and end. See the FiLM-path docstring.
      - `"adaln_zero"` (Phase C / v5+): AdaLN-Zero per-block modulation,
        DiT-style. ~13× more conditioning capacity per layer; identity at init.
    """
    regime_embed = RegimeEmbedding(
        n_categories_per_axis=n_categories_per_axis, embed_dim=embed_dim,
    )
    denoiser = TradesStyleDenoiser(
        n_features=n_features,
        d_model=d_model,
        num_heads=num_heads,
        depth=depth,
        max_seq_len=max_seq_len,
        ctx_dim=embed_dim,
        dropout=dropout,
        num_diffusionsteps=num_diffusionsteps,
        conditioning_type=conditioning_type,
    )
    return GeneratorModel(denoiser=denoiser, regime_embed=regime_embed)


def encode_window(df: pl.DataFrame, norm: NormStats | None = None) -> torch.Tensor:
    """Canonical Parquet window → (L, F_in) tensor in the denoiser's input space."""
    feats = df.select(FEATURE_COLUMNS).fill_null(0.0).to_numpy().astype("float32")
    if norm is not None:
        feats = norm.normalize(feats).astype("float32")
    return torch.from_numpy(feats)


def load_pretrained_trades_checkpoint(checkpoint_path: str | Path):
    """Stub: the published TRADES checkpoint expects LOBSTER message-level data
    (LEN_ORDER=6 + 40-feature LOB snapshot per step), which we do not produce
    from TAQ NBBO. Reusing those weights would require materializing
    LOBSTER-format messages from TAQ — out of scope. Use build_generator()
    above and fine-tune from scratch on our cleaned tapes.
    """
    raise NotImplementedError(
        "Pretrained TRADES checkpoint is incompatible with our TAQ-derived schema. "
        "Use build_generator() and train from scratch on cleaned TAQ tapes."
    )
