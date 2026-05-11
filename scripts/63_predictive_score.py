#!/usr/bin/env python3
"""
Predictive score (TRADES Table 1) — train a small LSTM next-mid_return predictor
on synthetic data, evaluate on real validation data. Compare across multiple
synthetic generators against a "market replay" baseline (train on real, test on real).

The premise: a good synthetic generator should be useful — i.e., a model trained
on it should generalize to real data. MAE on real data, normalized by the
market_replay MAE, is the predictive score.

Usage:
    python scripts/63_predictive_score.py \
        --real-train-dir data/processed/train \
        --real-test-dir data/processed/val \
        --synth-dirs \
            $HOME/orcd/scratch/diffmm/synthetic_v2_stitched/INTC \
            $HOME/orcd/scratch/diffmm/synthetic/intc_v9/INTC \
            $HOME/orcd/scratch/diffmm/synthetic_v2_remapped/INTC \
        --labels v2 v9 v2_remapped \
        --out results/predictive_score \
        --window-size 64 \
        --max-tapes-train 200 \
        --max-tapes-test 10 \
        --seed 42

CPU is fine (small LSTM, ~1M params). Runtime: ~5-10 min per generator.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 8 features the diffusion model uses (canonical schema column names)
FEATURE_COLS = [
    "mid_return", "spread", "bid_sz", "ask_sz",
    "bid_dist", "ask_dist", "trade_sz", "trade_sign",
]


# ─── Model ──────────────────────────────────────────────────────────────────

class MidPricePredictor(nn.Module):
    """1-layer LSTM that predicts next mid_return from a window of LOB features."""
    def __init__(self, input_dim: int = 8, hidden_dim: int = 64, n_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_dim)
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :]).squeeze(-1)   # (batch,)


# ─── Data loading & windowing ───────────────────────────────────────────────

def _list_parquets(directory: Path, max_n: int) -> list[Path]:
    """Return up to max_n parquets, recursing one level if needed."""
    direct = sorted(directory.glob("*.parquet"))
    if direct:
        return direct[:max_n]
    nested: list[Path] = []
    for sub in sorted(directory.iterdir()):
        if sub.is_dir():
            nested.extend(sorted(sub.glob("*.parquet")))
            if len(nested) >= max_n:
                break
    return nested[:max_n]


def load_features_per_tape(directory: str | Path, max_tapes: int) -> list[np.ndarray]:
    """Load each parquet into a (T, 8) float32 array. Tapes are kept SEPARATE
    so windows don't cross tape boundaries."""
    p = Path(directory)
    if not p.exists():
        raise FileNotFoundError(directory)
    paths = _list_parquets(p, max_tapes)
    if not paths:
        raise FileNotFoundError(f"no parquets in {directory}")
    out: list[np.ndarray] = []
    for pq in paths:
        try:
            df = pd.read_parquet(pq, columns=FEATURE_COLS)
        except Exception as e:
            print(f"  WARN: skipping {pq}: {e}", file=sys.stderr)
            continue
        # fill_null(0.0) — same convention as the diffmm dataset loader
        arr = df.fillna(0.0).to_numpy().astype(np.float32)
        if arr.shape[0] < 100:
            continue
        out.append(arr)
    if not out:
        raise FileNotFoundError(f"no usable tapes in {directory}")
    return out


def make_windows(tapes: list[np.ndarray], window: int, stride: int = 8,
                 max_windows: int | None = None, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Slide a length-`window` window across each tape with the given stride
    (no boundary crossing). Input X = features at events i..i+window-1;
    target y = mid_return at i+window.

    Default stride=8 balances dataset size across stitched (~4096-event) and
    raw (~256-event) tapes: gives ~512 windows per stitched tape and ~24 per
    raw tape, so 200 stitched tapes ≈ 100k windows, 200 raw tapes ≈ 5k windows.

    `max_windows`: if set, subsamples (tape_idx, start) PAIRS before materializing
    slices. This is the memory bound — without it, real-data tapes (~3M events
    × 120 tapes / stride 8 ≈ 27M windows × 64 × 8 × 4 bytes ≈ 52 GB) OOM the
    np.stack call. Subsampling the index list keeps memory at O(max_windows).
    """
    s_eff = max(int(stride), 1)
    # Pass 1: enumerate (tape_idx, start) pairs without allocating slice tensors.
    pairs: list[tuple[int, int]] = []
    for ti, arr in enumerate(tapes):
        T = arr.shape[0]
        if T <= window:
            continue
        for s in range(0, T - window, s_eff):
            pairs.append((ti, s))
    if not pairs:
        return np.empty((0, window, len(FEATURE_COLS)), dtype=np.float32), np.empty(0, dtype=np.float32)
    # Subsample BEFORE materializing.
    if max_windows is not None and len(pairs) > max_windows:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(pairs), size=max_windows, replace=False)
        pairs = [pairs[i] for i in idx]
    # Pass 2: allocate output arrays directly and copy selected windows.
    n = len(pairs)
    Xs = np.empty((n, window, len(FEATURE_COLS)), dtype=np.float32)
    ys = np.empty(n, dtype=np.float32)
    for i, (ti, s) in enumerate(pairs):
        Xs[i] = tapes[ti][s:s + window]
        ys[i] = tapes[ti][s + window, 0]   # mid_return is index 0 in FEATURE_COLS
    return Xs, ys


# ─── Normalization (training-set z-score, applied to test) ──────────────────

def cap_windows(X: np.ndarray, y: np.ndarray, max_windows: int, seed: int = 42,
                label: str = "") -> tuple[np.ndarray, np.ndarray]:
    """Random-subsample (X, y) to at most `max_windows` rows. No-op if already smaller.

    Memory bound: at default 200k windows × 64 × 8 × 4 bytes ≈ 3 GB per source.
    Without this cap, real-data sources (3M events × 120 tapes / stride 8 ≈
    27M windows ≈ 55 GB) OOM the LSTM training pipeline.
    """
    n = X.shape[0]
    if n <= max_windows:
        return X, y
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=max_windows, replace=False)
    print(f"  {label}: subsampling {n} → {max_windows} windows (memory cap)")
    return X[idx], y[idx]


def fit_norm(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-feature mean/std on the (N, T, F) training tensor."""
    flat = X.reshape(-1, X.shape[-1])
    mean = flat.mean(axis=0)
    std = flat.std(axis=0) + 1e-6
    return mean.astype(np.float32), std.astype(np.float32)


def apply_norm(X: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return ((X - mean) / std).astype(np.float32)


# ─── Train/eval ─────────────────────────────────────────────────────────────

class WindowDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)
    def __len__(self) -> int: return self.X.shape[0]
    def __getitem__(self, i: int):
        return self.X[i], self.y[i]


def train_one_model(
    label: str, train_X: np.ndarray, train_y: np.ndarray,
    test_X: np.ndarray, test_y: np.ndarray,
    *, batch_size: int = 256, max_epochs: int = 20, patience: int = 3,
    lr: float = 1e-3, device: str = "cpu", seed: int = 42,
    hidden_dim: int = 64,
) -> dict:
    """Fit on train_X/y (with internal 90/10 train/val split for early stop),
    evaluate MAE on test_X/y. Returns dict with test_mae + train metadata."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    n = train_X.shape[0]
    if n < 100:
        print(f"  WARN [{label}]: only {n} training windows — skipping")
        return {"label": label, "test_mae": float("nan"), "train_epochs": 0,
                "train_final_loss": float("nan"), "n_train_windows": n}

    # 90/10 split for early stopping
    perm = np.random.permutation(n)
    n_val = max(1, int(n * 0.1))
    val_idx = perm[:n_val]; train_idx = perm[n_val:]
    Xtr, ytr = train_X[train_idx], train_y[train_idx]
    Xva, yva = train_X[val_idx], train_y[val_idx]

    train_loader = DataLoader(WindowDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(WindowDataset(Xva, yva), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(WindowDataset(test_X, test_y), batch_size=batch_size, shuffle=False)

    model = MidPricePredictor(input_dim=train_X.shape[-1],
                              hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    mae_loss = nn.L1Loss()

    best_val_mae = float("inf")
    bad_epochs = 0
    last_loss = float("nan")
    epochs_done = 0
    for epoch in range(max_epochs):
        model.train()
        train_losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = mse_loss(pred, yb)
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        last_loss = float(np.mean(train_losses)) if train_losses else float("nan")

        # Val
        model.eval()
        val_maes = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                val_maes.append(mae_loss(pred, yb).item())
        val_mae = float(np.mean(val_maes)) if val_maes else float("inf")
        epochs_done = epoch + 1
        print(f"  [{label}] epoch {epoch+1:2d}  train_mse={last_loss:.4e}  val_mae={val_mae:.4e}")
        if val_mae < best_val_mae - 1e-8:
            best_val_mae = val_mae
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"  [{label}] early stop after {epoch+1} epochs (val MAE not improving)")
                break

    # Test MAE on REAL val data
    model.eval()
    test_maes = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            test_maes.append(mae_loss(pred, yb).item())
    test_mae = float(np.mean(test_maes)) if test_maes else float("nan")
    print(f"  [{label}] TEST MAE on real val = {test_mae:.4e}")

    return {
        "label": label,
        "test_mae": test_mae,
        "train_epochs": epochs_done,
        "train_final_loss": last_loss,
        "n_train_windows": int(n),
        "n_test_windows": int(test_X.shape[0]),
    }


# ─── Main ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--real-train-dir", required=True)
    ap.add_argument("--real-test-dir", required=True)
    ap.add_argument("--synth-dirs", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", required=True)
    ap.add_argument("--out", default="results/predictive_score")
    ap.add_argument("--window-size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=8,
                    help="Stride for sliding windows (default 8). Smaller → more "
                         "windows but more correlated samples; larger → fewer but "
                         "independent. Default 8 keeps raw 256-event tapes viable.")
    ap.add_argument("--max-tapes-train", type=int, default=200)
    ap.add_argument("--max-tapes-test", type=int, default=10)
    ap.add_argument("--max-windows-per-source", type=int, default=200_000,
                    help="Random-subsample windows per training source after "
                         "window construction. Default 200k — bounds memory at "
                         "~3 GB per source (200k × 64 × 8 × 4 bytes). Real-data "
                         "tapes have ~3M events each, which at stride=8 produces "
                         "~400k windows per tape; without this cap the market_replay "
                         "baseline OOMs. A small LSTM doesn't need more than ~100k samples.")
    ap.add_argument("--max-windows-test", type=int, default=100_000,
                    help="Random-subsample test windows. Default 100k.")
    ap.add_argument("--max-epochs", type=int, default=20)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--hidden-dim", type=int, default=64,
                    help="LSTM hidden dim. Use a sweep over {32,64,128} to "
                         "test ranking robustness (P2.G4 in Work10_update.md).")
    args = ap.parse_args()

    if len(args.synth_dirs) != len(args.labels):
        raise SystemExit("--synth-dirs and --labels must have same count")

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("  WARN: cuda requested but unavailable; using cpu")
        device = "cpu"

    os.makedirs(args.out, exist_ok=True)

    # ── Build the test set ONCE (real validation data) ──
    print(f"\nLoading real TEST tapes from {args.real_test_dir} ...")
    real_test_tapes = load_features_per_tape(args.real_test_dir, args.max_tapes_test)
    print(f"  {len(real_test_tapes)} real test tapes loaded")
    real_test_X, real_test_y = make_windows(real_test_tapes, args.window_size, args.stride,
                                              max_windows=args.max_windows_test, seed=args.seed)
    print(f"  {real_test_X.shape[0]} test windows ({args.window_size}-event each, capped at {args.max_windows_test})")

    # ── Market replay baseline: train on real train, test on real val ──
    print(f"\nLoading real TRAIN tapes (market_replay baseline) from {args.real_train_dir} ...")
    real_train_tapes = load_features_per_tape(args.real_train_dir, args.max_tapes_train)
    print(f"  {len(real_train_tapes)} real train tapes loaded")
    real_train_X, real_train_y = make_windows(real_train_tapes, args.window_size, args.stride,
                                                max_windows=args.max_windows_per_source, seed=args.seed)
    print(f"  {real_train_X.shape[0]} train windows (capped at {args.max_windows_per_source})")

    results: list[dict] = []
    t0 = time.time()

    print("\n=== Training market_replay baseline (train: real_train, test: real_val) ===")
    mean_r, std_r = fit_norm(real_train_X)
    Xtr_n = apply_norm(real_train_X, mean_r, std_r)
    Xte_n = apply_norm(real_test_X, mean_r, std_r)
    res = train_one_model("market_replay", Xtr_n, real_train_y, Xte_n, real_test_y,
                          batch_size=args.batch_size, max_epochs=args.max_epochs,
                          patience=args.patience, lr=args.lr, device=device, seed=args.seed,
                          hidden_dim=args.hidden_dim)
    res["train_source"] = "real_train"
    results.append(res)

    replay_mae = res["test_mae"] if np.isfinite(res["test_mae"]) else float("nan")

    # ── Each synth model: train on synth, test on real_val ──
    for sdir, label in zip(args.synth_dirs, args.labels):
        print(f"\n=== Training on {label}  (synth_dir: {sdir}) ===")
        try:
            synth_tapes = load_features_per_tape(sdir, args.max_tapes_train)
        except FileNotFoundError as e:
            print(f"  SKIP {label}: {e}")
            continue
        print(f"  {len(synth_tapes)} synth tapes loaded")
        Xtr, ytr = make_windows(synth_tapes, args.window_size, args.stride,
                                  max_windows=args.max_windows_per_source, seed=args.seed)
        print(f"  {Xtr.shape[0]} train windows from {label} (capped at {args.max_windows_per_source})")
        if Xtr.shape[0] < 100:
            print(f"  SKIP {label}: too few windows ({Xtr.shape[0]})")
            continue

        # Per-source training-set normalization
        mean_s, std_s = fit_norm(Xtr)
        Xtr_n = apply_norm(Xtr, mean_s, std_s)
        Xte_n = apply_norm(real_test_X, mean_s, std_s)
        res = train_one_model(label, Xtr_n, ytr, Xte_n, real_test_y,
                              batch_size=args.batch_size, max_epochs=args.max_epochs,
                              patience=args.patience, lr=args.lr, device=device, seed=args.seed,
                              hidden_dim=args.hidden_dim)
        res["train_source"] = label + "_synth"
        results.append(res)

    # ── Compute relative-to-replay ratio ──
    for r in results:
        if np.isfinite(replay_mae) and replay_mae > 0:
            r["relative_to_replay"] = r["test_mae"] / replay_mae
        else:
            r["relative_to_replay"] = float("nan")

    # ── Output: CSV + table + figure ──
    csv_path = f"{args.out}/predictive_score.csv"
    df_results = pd.DataFrame(results)[
        ["label", "train_source", "test_mae", "relative_to_replay",
         "train_epochs", "train_final_loss", "n_train_windows", "n_test_windows"]
    ]
    df_results.to_csv(csv_path, index=False)

    print("\n" + "=" * 80)
    print("PREDICTIVE SCORE — TRADES Table 1")
    print("=" * 80)
    print(f"\n{'Model':<18s} {'Train Source':<15s} {'Test MAE':>14s} {'Relative to Replay':>22s}")
    print("-" * 75)
    for r in results:
        rel = r["relative_to_replay"]
        rel_str = f"{rel:>20.2f}×" if np.isfinite(rel) else f"{'-':>22s}"
        mae_str = f"{r['test_mae']:>14.4e}" if np.isfinite(r['test_mae']) else f"{'-':>14s}"
        print(f"{r['label']:<18s} {r['train_source']:<15s} {mae_str} {rel_str}")
    print()
    print(f"CSV: {csv_path}")
    print(f"Wall time: {time.time() - t0:.1f}s")

    # ── Bar chart ──
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    labels = [r["label"] for r in results if np.isfinite(r["test_mae"])]
    maes = [r["test_mae"] for r in results if np.isfinite(r["test_mae"])]
    cmap = plt.get_cmap("tab10")
    bars = ax.barh(labels, maes,
                    color=["0.4" if l == "market_replay" else cmap(i % 10)
                           for i, l in enumerate(labels)])
    if np.isfinite(replay_mae):
        ax.axvline(replay_mae, color="0.4", ls="--", lw=1, label=f"market_replay = {replay_mae:.3e}")
    ax.set_xlabel("Test MAE on real validation data (lower is better)")
    ax.set_title("Predictive Score — train on each source, test on real val")
    ax.legend(loc="best", fontsize=9)
    ax.grid(True, alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(f"{args.out}/predictive_score.pdf")
    fig.savefig(f"{args.out}/predictive_score.png", dpi=120)
    plt.close(fig)
    print(f"Figure: {args.out}/predictive_score.pdf")


if __name__ == "__main__":
    main()
