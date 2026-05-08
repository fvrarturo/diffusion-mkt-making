#!/usr/bin/env python3
"""
Experiment 1 (Phase E coda, 2026-05-08) — Post-hoc quantile remap of v2's
synthetic parquets to match real INTC's per-feature marginals.

Mathematically:
    x_remapped_j = F̂_real_j⁻¹( F̂_v2_j(x_v2_j) )    for each feature j

Spearman rank correlations between feature pairs are preserved EXACTLY
(monotone-invariance). Temporal rank-autocorrelation is preserved exactly.
Whatever fill-rate dynamics drove v2's ρ_diff = +0.40 operate on the rank
ordering of events, which is unchanged.

What changes: marginals become exact matches to real. mid_return gains
real's heavy-tailed kurtosis. trade_sz gains real's variance. frac_zero
goes from ~0 to ~0.89.

The critical question: can we have v2's joints AND real's marginals, or
does the absolute magnitude matter for fill dynamics?

Outputs: a parallel parquet directory (canonical schema) ready to be fed
into ph4_replay_synth + ph6_validate + ph5_hypothesis exactly like v2 was.

Usage (CPU, runs in minutes on existing v2 parquets):
    python scripts/70_remap_v2.py \
        --source-dir $HOME/orcd/scratch/diffmm/synthetic_v2_stitched/INTC \
        --reference-dir data/processed/val \
        --out-dir $HOME/orcd/scratch/diffmm/synthetic_v2_remapped/INTC \
        --norm-stats data/processed/norm_stats_INTC.json \
        --max-source-events 2000000 \
        --max-reference-events 2000000

Then run the standard pipeline pointing at the remapped output:
    sbatch slurms/ph4_replay_synth.slurm \
        data=intc_top \
        +synthetic_root=$HOME/orcd/scratch/diffmm/synthetic_v2_remapped
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

# Avoid hard dependency on the heavier scipy.interpolate at import time;
# fall back to numpy.interp (1-D linear) which is sufficient and faster.
from diffmm.data.dataset import FEATURE_COLUMNS

# Tick size convention from src/diffmm/generator/decode.py
DEFAULT_TICK_SIZE_USD = 0.01


# ─────────────────────────────────────────────────────────────────────────────
# Quantile map
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class QuantileMap:
    """Monotone interp1d-style mapping from source quantiles to target quantiles."""
    src_q: np.ndarray  # sorted source quantiles
    tgt_q: np.ndarray  # sorted target quantiles

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # np.interp clamps outside [src_q[0], src_q[-1]] to the boundary tgt_q values.
        return np.interp(x, self.src_q, self.tgt_q)


def fit_quantile_map(source_vals: np.ndarray, target_vals: np.ndarray,
                     n_quantiles: int = 10_000) -> QuantileMap:
    """Build the monotone empirical-CDF map source → target."""
    qs = np.linspace(0.0, 1.0, n_quantiles)
    src_q = np.quantile(source_vals.astype(np.float64), qs)
    tgt_q = np.quantile(target_vals.astype(np.float64), qs)
    # np.interp requires a strictly-increasing src_q. Tied source quantiles
    # (e.g. v2 outputs many identical values) need to be handled by adding
    # tiny ε that monotonically increases. This preserves the mapping shape
    # without changing the target values.
    if not np.all(np.diff(src_q) > 0):
        # Add monotonic jitter scaled to feature range
        scale = max(float(src_q[-1] - src_q[0]), 1e-12) * 1e-12
        src_q = src_q + np.arange(len(src_q)) * scale
    return QuantileMap(src_q=src_q, tgt_q=tgt_q)


# ─────────────────────────────────────────────────────────────────────────────
# Loading helpers — sample-bounded to keep memory reasonable
# ─────────────────────────────────────────────────────────────────────────────


def _list_parquets(directory: Path) -> list[Path]:
    """Find parquet files in `directory` recursively (one level of regime subdirs)."""
    direct = sorted(directory.glob("*.parquet"))
    if direct:
        return direct
    nested: list[Path] = []
    for sub in sorted(directory.iterdir()):
        if sub.is_dir():
            nested.extend(sorted(sub.glob("*.parquet")))
    return nested


def pool_feature_values(parquets: list[Path], max_events: int,
                        rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Concatenate FEATURE_COLUMNS values across parquets, with reservoir-style
    subsampling to cap total memory at `max_events` per feature."""
    chunks_per_feature: dict[str, list[np.ndarray]] = {c: [] for c in FEATURE_COLUMNS}
    total = 0
    for p in parquets:
        try:
            df = pl.read_parquet(p, columns=list(FEATURE_COLUMNS))
        except Exception as e:
            print(f"  warn: skipping {p}: {e}", file=sys.stderr)
            continue
        # fill_null(0.0) so trade_sz/trade_dist/trade_sign on quote_update events
        # are treated as zeros — same convention as the dataset loader.
        arr = df.fill_null(0.0).to_numpy()
        n = arr.shape[0]
        for j, col in enumerate(FEATURE_COLUMNS):
            chunks_per_feature[col].append(arr[:, j].astype(np.float64))
        total += n
        if total >= max_events:
            break
    pooled: dict[str, np.ndarray] = {}
    for col, chunks in chunks_per_feature.items():
        if not chunks:
            pooled[col] = np.empty(0, dtype=np.float64)
            continue
        cat = np.concatenate(chunks)
        if len(cat) > max_events:
            idx = rng.choice(len(cat), size=max_events, replace=False)
            cat = cat[idx]
        pooled[col] = cat
    return pooled


# ─────────────────────────────────────────────────────────────────────────────
# Per-parquet remap + re-derive dependent columns
# ─────────────────────────────────────────────────────────────────────────────


def reconstruct_canonical(features: dict[str, np.ndarray], anchor_mid: float,
                          template: pl.DataFrame, tick_size: float = DEFAULT_TICK_SIZE_USD
                          ) -> pl.DataFrame:
    """Re-derive bid_px/ask_px/mid/spread/etc. from remapped 8-feature inputs.

    Mirrors `src/diffmm/generator/decode.py::decode_window_to_dataframe` exactly,
    minus the denormalization step (features are already in original scale).
    Keeps `event_idx`, `ts_ns`, `ticker`, `c_*`, `regime_label` from the original
    parquet as those are not affected by the remap.
    """
    L = next(iter(features.values())).shape[0]

    # ── mid trajectory from remapped mid_return (anchor + cumsum log-returns) ──
    mid_returns = features["mid_return"].copy()
    mid_returns[0] = 0.0  # decode convention
    mid = anchor_mid * np.exp(np.cumsum(mid_returns))
    mid = np.maximum(mid, tick_size)

    # ── bid/ask price reconstruction with crossed-book guard (mirrors decode.py) ──
    bid_dist = np.minimum(features["bid_dist"], 0.0)
    ask_dist = np.maximum(features["ask_dist"], 0.0)
    bid_px = mid + bid_dist
    ask_px = mid + ask_dist
    crossed = ask_px <= bid_px
    half_tick = 0.5 * tick_size
    bid_px = np.where(crossed, mid - half_tick, bid_px)
    ask_px = np.where(crossed, mid + half_tick, ask_px)
    bid_px = np.maximum(bid_px, tick_size)
    ask_px = np.maximum(ask_px, bid_px + tick_size)
    spread = ask_px - bid_px

    # ── sizes (round to int, floor at 0) ──
    bid_sz = np.maximum(np.rint(features["bid_sz"]), 0).astype(np.int64)
    ask_sz = np.maximum(np.rint(features["ask_sz"]), 0).astype(np.int64)

    # ── trades: nullable on no-trade events ──
    trade_sz_round = np.maximum(np.rint(features["trade_sz"]), 0).astype(np.int64)
    is_trade = trade_sz_round > 0
    trade_dist_arr = features["trade_dist"]
    trade_px_arr = mid + trade_dist_arr
    trade_sign_raw = np.sign(features["trade_sign"])
    trade_sign_inferred = np.where(trade_dist_arr >= 0, 1, -1)
    trade_sign_arr = np.where(
        is_trade,
        np.where(trade_sign_raw == 0, trade_sign_inferred, trade_sign_raw),
        0,
    ).astype(np.int8)

    # Rebuild nullable columns. Use lists with `None` so polars infers the
    # nullable types correctly (we'll cast explicitly below).
    trade_px_list = [float(v) if is_trade[i] else None for i, v in enumerate(trade_px_arr)]
    trade_sz_list = [int(v) if is_trade[i] else None for i, v in enumerate(trade_sz_round)]
    trade_sign_list = [int(v) if is_trade[i] else None for i, v in enumerate(trade_sign_arr)]
    trade_dist_list = [float(v) if is_trade[i] else None for i, v in enumerate(trade_dist_arr)]
    is_lit_list = [True if is_trade[i] else None for i in range(L)]
    event_type = np.where(is_trade, "trade", "quote_update").astype(object)

    mid_return_canonical = [None] + np.diff(np.log(mid)).tolist()

    # ── Build the new DataFrame, keeping event_idx/ts_ns/conditions from template ──
    new_cols = {
        "event_idx":    template["event_idx"].to_numpy(),
        "ts_ns":        template["ts_ns"].to_numpy(),
        "ticker":       template["ticker"].to_list(),
        "event_type":   list(event_type),
        "bid_px":       bid_px,
        "ask_px":       ask_px,
        "bid_sz":       bid_sz,
        "ask_sz":       ask_sz,
        "trade_px":     trade_px_list,
        "trade_sz":     trade_sz_list,
        "trade_sign":   trade_sign_list,
        "is_lit":       is_lit_list,
        "mid":          mid,
        "spread":       spread,
        "bid_dist":     (bid_px - mid),
        "ask_dist":     (ask_px - mid),
        "trade_dist":   trade_dist_list,
        "mid_return":   mid_return_canonical,
        "c_vol":        template["c_vol"].to_numpy(),
        "c_vpin":       template["c_vpin"].to_numpy(),
        "c_imb":        template["c_imb"].to_numpy(),
        "c_tod":        template["c_tod"].to_numpy(),
        "regime_label": template["regime_label"].to_list(),
    }
    df = pl.DataFrame(new_cols)
    df = df.with_columns([
        pl.col("ticker").cast(pl.Categorical),
        pl.col("event_type").cast(pl.Categorical),
        pl.col("trade_sign").cast(pl.Int8),
        pl.col("regime_label").cast(pl.Categorical),
    ])
    # Null-column recast — same dance as decode.py for empty-trade windows
    nullable_recasts = {
        "trade_px": pl.Float64, "trade_sz": pl.Int64,
        "trade_sign": pl.Int8, "is_lit": pl.Boolean,
        "trade_dist": pl.Float64, "mid_return": pl.Float64,
    }
    cast_exprs = [
        pl.col(col).cast(dtype)
        for col, dtype in nullable_recasts.items()
        if col in df.columns and df.schema[col] != dtype
    ]
    if cast_exprs:
        df = df.with_columns(cast_exprs)
    return df


def remap_one_parquet(in_path: Path, out_path: Path,
                      qmaps: dict[str, QuantileMap], anchor_mid: float,
                      tick_size: float = DEFAULT_TICK_SIZE_USD) -> int:
    """Apply per-feature remap to one parquet, re-derive dependent columns,
    write to out_path. Returns number of events processed."""
    df = pl.read_parquet(in_path)
    arr = df.select(list(FEATURE_COLUMNS)).fill_null(0.0).to_numpy().astype(np.float64)
    remapped = {col: qmaps[col](arr[:, j]) for j, col in enumerate(FEATURE_COLUMNS)}
    new_df = reconstruct_canonical(remapped, anchor_mid=anchor_mid,
                                    template=df, tick_size=tick_size)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_df.write_parquet(out_path)
    return arr.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source-dir", required=True,
                    help="Directory of source synthetic parquets (e.g. v2 stitched, with regime subdirs)")
    ap.add_argument("--reference-dir", required=True,
                    help="Directory of real reference parquets (e.g. data/processed/val)")
    ap.add_argument("--out-dir", required=True,
                    help="Output directory (mirrors source-dir's subdir structure)")
    ap.add_argument("--norm-stats", default="data/processed/norm_stats_INTC.json",
                    help="Path to NormStats JSON (used to read anchor_mid for mid integration)")
    ap.add_argument("--features", nargs="+", default=None,
                    help="Subset of features to remap. Others use the identity map "
                         "(source values flow through unchanged). Default: all FEATURE_COLUMNS. "
                         "Useful for isolating which marginal change drives downstream effects: "
                         "e.g. --features mid_return remaps ONLY mid_return and keeps v2's "
                         "size/trade distributions intact (preserves the trade/quote event mix).")
    ap.add_argument("--max-source-events", type=int, default=2_000_000,
                    help="Max events sampled from source for quantile pool (per feature)")
    ap.add_argument("--max-reference-events", type=int, default=5_000_000,
                    help="Max events sampled from reference for quantile pool (per feature)")
    ap.add_argument("--n-quantiles", type=int, default=10_000,
                    help="Number of quantile points for the empirical CDF maps")
    ap.add_argument("--tick-size", type=float, default=DEFAULT_TICK_SIZE_USD)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    src_dir = Path(args.source_dir)
    ref_dir = Path(args.reference_dir)
    out_dir = Path(args.out_dir)
    if not src_dir.exists():
        raise SystemExit(f"source-dir not found: {src_dir}")
    if not ref_dir.exists():
        raise SystemExit(f"reference-dir not found: {ref_dir}")

    # Anchor mid for mid-trajectory integration. Re-use whatever the original
    # v2 used (NormStats anchor_mid is the training-set median).
    import json
    with open(args.norm_stats) as f:
        norm_meta = json.load(f)
    anchor_mid = float(norm_meta.get("anchor_mid", 0.0))
    if anchor_mid <= 0:
        raise SystemExit(f"invalid anchor_mid={anchor_mid} from {args.norm_stats}")
    print(f"[remap] anchor_mid = {anchor_mid:.4f}")

    # ── Pool source values ──
    src_parquets = _list_parquets(src_dir)
    if not src_parquets:
        raise SystemExit(f"no parquets in {src_dir}")
    print(f"[remap] source: {len(src_parquets)} parquets in {src_dir}")
    src_pool = pool_feature_values(src_parquets, args.max_source_events, rng)
    for col, vals in src_pool.items():
        print(f"  src   {col:12s}  n={len(vals):>9d}  mean={np.mean(vals):+.6g}  std={np.std(vals):.6g}")

    # ── Pool reference (real) values ──
    ref_parquets = _list_parquets(ref_dir)
    if not ref_parquets:
        raise SystemExit(f"no parquets in {ref_dir}")
    print(f"[remap] reference: {len(ref_parquets)} parquets in {ref_dir}")
    ref_pool = pool_feature_values(ref_parquets, args.max_reference_events, rng)
    for col, vals in ref_pool.items():
        print(f"  ref   {col:12s}  n={len(vals):>9d}  mean={np.mean(vals):+.6g}  std={np.std(vals):.6g}")

    # ── Fit per-feature quantile maps ──
    # If --features is set, only remap those columns; others get an identity map
    # (source values pass through unchanged). Lets us isolate which marginal change
    # drives downstream effects: e.g. --features mid_return preserves v2's
    # trade/quote event mix while only fixing the heavy-tail return failure.
    if args.features:
        selected = set(args.features)
        invalid = selected - set(FEATURE_COLUMNS)
        if invalid:
            raise SystemExit(f"--features contains unknown columns: {invalid}. "
                             f"Valid: {FEATURE_COLUMNS}")
        print(f"[remap] PARTIAL remap — features to remap: {sorted(selected)}")
        print(f"        identity (passthrough) for: {[c for c in FEATURE_COLUMNS if c not in selected]}")
    else:
        selected = set(FEATURE_COLUMNS)
    qmaps: dict[str, QuantileMap] = {}
    for col in FEATURE_COLUMNS:
        if col not in selected:
            # Identity map: y = x (build trivial 2-point map covering the full real range).
            lo = float(np.min(src_pool[col])) if len(src_pool[col]) else -1e10
            hi = float(np.max(src_pool[col])) if len(src_pool[col]) else 1e10
            qmaps[col] = QuantileMap(src_q=np.array([lo, hi]), tgt_q=np.array([lo, hi]))
            print(f"  pass  {col:12s}  (identity)")
            continue
        if len(src_pool[col]) == 0 or len(ref_pool[col]) == 0:
            print(f"  WARN: missing data for {col}; using identity map")
            qmaps[col] = QuantileMap(src_q=np.array([-1e10, 1e10]),
                                     tgt_q=np.array([-1e10, 1e10]))
            continue
        qmaps[col] = fit_quantile_map(src_pool[col], ref_pool[col],
                                      n_quantiles=args.n_quantiles)
        # Quick sanity check on the map endpoints
        print(f"  map   {col:12s}  src[1%, 99%]=[{np.quantile(src_pool[col], 0.01):+.6g}, "
              f"{np.quantile(src_pool[col], 0.99):+.6g}]  ->  "
              f"tgt[1%, 99%]=[{np.quantile(ref_pool[col], 0.01):+.6g}, "
              f"{np.quantile(ref_pool[col], 0.99):+.6g}]")

    # ── Apply to every source parquet ──
    print(f"[remap] writing remapped parquets to {out_dir}")
    n_done = 0
    n_events_total = 0
    for in_path in src_parquets:
        rel = in_path.relative_to(src_dir)
        out_path = out_dir / rel
        try:
            n_events = remap_one_parquet(in_path, out_path, qmaps, anchor_mid, args.tick_size)
            n_events_total += n_events
            n_done += 1
            if n_done % 200 == 0:
                print(f"  remapped {n_done}/{len(src_parquets)} parquets "
                      f"({n_events_total:,} events)")
        except Exception as e:
            print(f"  ERROR remapping {in_path}: {e}", file=sys.stderr)

    print(f"[remap] DONE — {n_done} parquets, {n_events_total:,} events")
    print(f"[remap] output at {out_dir}")
    print()
    print("Next step: run ph4_replay_synth pointing at the remapped parquets, e.g.:")
    print(f"  sbatch slurms/ph4_replay_synth.slurm \\")
    print(f"      data=intc_top \\")
    print(f"      +synthetic_root={out_dir.parent}")


if __name__ == "__main__":
    main()
