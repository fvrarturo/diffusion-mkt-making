"""
84_regime_embedding_tsne.py — Visualise the learned regime embeddings of one
or more checkpoints in 2-D (t-SNE / UMAP / PCA fallback). Lets us see whether
the four regime indices (base / high_vol / toxic / thin) actually map to
distinct points in embedding space, and how that distinctness varies across
architectures (FiLM vs AdaLN-Zero is the key contrast).

Reads:  Lightning checkpoint files (.ckpt)
Writes: results/report_figures/fig_p2_4_regime_tsne.{pdf,png}
        results/report_figures/regime_embedding_distances.csv

Run:
    python scripts/84_regime_embedding_tsne.py \
        --ckpts $HOME/orcd/scratch/diffmm/checkpoints/intc_v2/ckpt-best.ckpt:v2 \
                $HOME/orcd/scratch/diffmm/checkpoints/intc_v5/ckpt-best.ckpt:v5 \
                $HOME/orcd/scratch/diffmm/checkpoints/intc_v9/ckpt-best.ckpt:v9
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "results" / "report_figures"
OUT.mkdir(parents=True, exist_ok=True)

REGIMES = ["base", "high_vol", "toxic", "thin"]


def load_regime_embeddings(ckpt_path: Path) -> np.ndarray:
    """Pull just the RegimeEmbedding weights from a Lightning checkpoint.
    Returns an array of shape (4, embed_dim)."""
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state.get("state_dict", state)
    # Look for keys ending in regime_embed.embedding.weight, .embed.weight,
    # .table.weight — depending on which embedding wrapper was used.
    candidates = [k for k in sd.keys()
                  if "regime_embed" in k and k.endswith(".weight")
                  and "null" not in k]
    if not candidates:
        raise KeyError(f"no regime_embed weight found in {ckpt_path}; "
                       f"keys sample: {list(sd.keys())[:6]}")
    # Prefer the 4-row embedding (4 regimes); skip null-condition projections
    emb = None
    for k in candidates:
        w = sd[k].detach().cpu().numpy()
        if w.ndim == 2 and w.shape[0] == 4:
            emb = w
            break
    if emb is None:
        # fallback: take the first that's 2-D
        for k in candidates:
            w = sd[k].detach().cpu().numpy()
            if w.ndim == 2:
                emb = w
                break
    if emb is None:
        raise RuntimeError(f"could not extract a 2-D regime embedding from {ckpt_path}")
    return emb


def project_2d(X: np.ndarray) -> tuple[np.ndarray, str]:
    """Project (n, d) → (n, 2). Try t-SNE first, fall back to PCA."""
    n = X.shape[0]
    if n < 3:
        return X[:, :2] if X.shape[1] >= 2 else np.column_stack([X, np.zeros(n)]), "raw"
    try:
        from sklearn.manifold import TSNE
        # perplexity must be < n_samples
        perplexity = min(2.0, max(1.5, n - 1.0))
        tsne = TSNE(n_components=2, perplexity=perplexity,
                    init="pca", learning_rate="auto", random_state=42)
        return tsne.fit_transform(X), "t-SNE"
    except Exception as e:
        print(f"  t-SNE failed ({e}); falling back to PCA", file=sys.stderr)
        from sklearn.decomposition import PCA
        return PCA(n_components=2).fit_transform(X), "PCA"


def pairwise_cosine(X: np.ndarray) -> np.ndarray:
    n = X.shape[0]
    out = np.zeros((n, n))
    norms = np.linalg.norm(X, axis=1) + 1e-12
    for i in range(n):
        for j in range(n):
            out[i, j] = float(X[i] @ X[j]) / (norms[i] * norms[j])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True,
                    help="ckpt_path:label pairs, e.g. /path/to/v2.ckpt:v2")
    args = ap.parse_args()

    # Parse ckpt:label pairs
    items: list[tuple[Path, str]] = []
    for raw in args.ckpts:
        if ":" not in raw:
            sys.exit(f"each --ckpts entry must be path:label  (got: {raw})")
        path, label = raw.rsplit(":", 1)
        items.append((Path(path), label))

    sns.set_theme(style="whitegrid", context="paper")
    n = len(items)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 5), squeeze=False)
    rows: list[dict] = []

    for ax, (path, label) in zip(axes[0], items):
        try:
            emb = load_regime_embeddings(path)
        except Exception as e:
            print(f"[{label}] FAIL: {e}", file=sys.stderr)
            ax.set_visible(False)
            continue
        print(f"[{label}] embedding shape = {emb.shape}")
        proj, method = project_2d(emb)
        for i in range(min(4, len(proj))):
            ax.scatter(proj[i, 0], proj[i, 1], s=180,
                       label=REGIMES[i] if i < len(REGIMES) else f"r{i}",
                       edgecolors="k", linewidths=0.6, alpha=0.9)
            ax.annotate(REGIMES[i] if i < len(REGIMES) else f"r{i}",
                        (proj[i, 0], proj[i, 1]),
                        xytext=(6, 4), textcoords="offset points", fontsize=10)
        ax.set_title(f"{label}  ({method})")
        ax.legend(fontsize=8, loc="best")

        # Pairwise stats (cosine + L2) — write to CSV
        cos = pairwise_cosine(emb[:4])
        for i in range(min(4, len(emb))):
            for j in range(i + 1, min(4, len(emb))):
                rows.append({
                    "model": label,
                    "regime_i": REGIMES[i] if i < 4 else f"r{i}",
                    "regime_j": REGIMES[j] if j < 4 else f"r{j}",
                    "cosine": float(cos[i, j]),
                    "l2": float(np.linalg.norm(emb[i] - emb[j])),
                })

    fig.suptitle("Regime embedding projection per checkpoint\n"
                 "(distinct cluster = conditioning is differentiating regimes)",
                 fontsize=13, y=1.04)
    fig.tight_layout()
    pdf = OUT / "fig_p2_4_regime_tsne.pdf"
    png = OUT / "fig_p2_4_regime_tsne.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"wrote {pdf}")

    if rows:
        out_csv = OUT / "regime_embedding_distances.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"wrote {out_csv}")


if __name__ == "__main__":
    main()
