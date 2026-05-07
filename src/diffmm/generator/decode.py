"""Decode synthetic generator output into canonical event-tape Parquet.

The generator produces samples in price-invariant feature space:
FEATURE_COLUMNS = (bid_dist, ask_dist, bid_sz, ask_sz, trade_dist, trade_sz,
                   trade_sign, mid_return)

This module reconstructs absolute prices for the canonical Parquet output:
  1. Inverse-normalize via NormStats.
  2. Integrate mid_return forward from anchor_mid → mid trajectory.
  3. bid_px = mid + bid_dist  ;  ask_px = mid + ask_dist  ;  trade_px = mid + trade_dist.
  4. Enforce no-crossed-book and non-negativity invariants.
  5. Round sizes; infer event_type from generated trade_sz.
  6. Synthesize ts_ns, fill regime columns, compute spread.
  7. Validate against the canonical schema and write Parquet.

The output Parquet matches the canonical schema exactly (including the absolute
price columns the fill simulator needs), which is what makes downstream code
blind to provenance (Work0.md §3).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import polars as pl
import torch

from ..io import parquet, schema
from ..data.dataset import (
    FEATURE_COLUMNS, CONDITION_COLUMNS, NormStats,
)

DEFAULT_TICK_SIZE_USD = 0.01
DEFAULT_INTER_EVENT_NS = 100_000_000  # 100 ms uniform; replace once Δt is a learned channel

_REGIME_LABEL_TABLE: dict[tuple[int, int], str] = {
    (2, 2): "toxic",     (2, 0): "high_vol", (2, 1): "high_vol",
    (0, 0): "base",      (0, 1): "base",     (1, 0): "base", (1, 1): "base",
    (1, 2): "thin",      (0, 2): "thin",
}


def _regime_label(c_vol: int, c_vpin: int) -> str:
    return _REGIME_LABEL_TABLE.get((int(c_vol), int(c_vpin)), "base")


def decode_window_to_dataframe(
    sample: torch.Tensor,           # shape (L, F=8) in normalized relative-feature space
    *,
    condition: torch.Tensor,        # shape (n_axes,)  long
    ticker: str,
    start_ts_ns: int,
    norm_stats: NormStats,          # required — provides anchor_mid + denormalization
    tick_size_usd: float = DEFAULT_TICK_SIZE_USD,
    inter_event_ns: int = DEFAULT_INTER_EVENT_NS,
) -> pl.DataFrame:
    """Convert one synthetic (L, F) tensor to a canonical event-tape DataFrame."""
    if sample.dim() != 2:
        raise ValueError(f"expected (L, F), got shape {tuple(sample.shape)}")
    if sample.shape[1] != len(FEATURE_COLUMNS):
        raise ValueError(
            f"sample has {sample.shape[1]} channels, expected {len(FEATURE_COLUMNS)} "
            f"({FEATURE_COLUMNS})"
        )

    x = sample.detach().cpu().numpy().astype(np.float64)
    x = norm_stats.denormalize(x)

    # Unpack channels by name — matches FEATURE_COLUMNS ordering.
    chan = {name: x[:, i] for i, name in enumerate(FEATURE_COLUMNS)}
    L = x.shape[0]

    # Reconstruct absolute prices. Integrate mid_return → mid trajectory
    # anchored at anchor_mid. Convention: chan["mid_return"][0] is the (null
    # in training) first-event placeholder; treat as 0 to avoid an initial jump.
    mid_returns = chan["mid_return"].copy()
    mid_returns[0] = 0.0
    mid = norm_stats.anchor_mid * np.exp(np.cumsum(mid_returns))
    mid = np.maximum(mid, tick_size_usd)        # numerical floor

    # Apply minimum-distortion clamps. The model's expected output is
    # bid_dist ≤ 0, ask_dist ≥ 0, ask_dist - bid_dist ≥ tick. Only correct
    # when the model violates one of these; do not "recompute mid" afterward,
    # which would silently drift mid away from the integrated value.
    bid_dist = np.minimum(chan["bid_dist"], 0.0)        # bid at-or-below mid
    ask_dist = np.maximum(chan["ask_dist"], 0.0)        # ask at-or-above mid
    bid_px = mid + bid_dist
    ask_px = mid + ask_dist
    # No-crossed-book guard: if the model produced an inverted or zero spread,
    # widen symmetrically by half a tick around mid. Otherwise leave both
    # sides at their generated distances.
    crossed = ask_px <= bid_px
    half_tick = 0.5 * tick_size_usd
    bid_px = np.where(crossed, mid - half_tick, bid_px)
    ask_px = np.where(crossed, mid + half_tick, ask_px)
    # Numerical floor on prices (no negatives even after extreme drift).
    bid_px = np.maximum(bid_px, tick_size_usd)
    ask_px = np.maximum(ask_px, bid_px + tick_size_usd)
    spread = ask_px - bid_px

    bid_sz = np.maximum(np.rint(chan["bid_sz"]), 0).astype(np.int64)
    ask_sz = np.maximum(np.rint(chan["ask_sz"]), 0).astype(np.int64)

    # Trades.
    trade_sz_round = np.maximum(np.rint(chan["trade_sz"]), 0).astype(np.int64)
    is_trade = trade_sz_round > 0
    trade_dist_arr = chan["trade_dist"]
    trade_px_arr = mid + trade_dist_arr                  # absolute trade price
    trade_sign_raw = np.sign(chan["trade_sign"])
    trade_sign_inferred = np.where(trade_dist_arr >= 0, 1, -1)
    trade_sign_arr = np.where(
        is_trade,
        np.where(trade_sign_raw == 0, trade_sign_inferred, trade_sign_raw),
        0,
    ).astype(np.int8)

    trade_px_list = [float(v) if is_trade[i] else None for i, v in enumerate(trade_px_arr)]
    trade_sz_list = [int(v) if is_trade[i] else None for i, v in enumerate(trade_sz_round)]
    trade_sign_list = [int(v) if is_trade[i] else None for i, v in enumerate(trade_sign_arr)]
    trade_dist_list = [float(v) if is_trade[i] else None for i, v in enumerate(trade_dist_arr)]
    is_lit_list = [True if is_trade[i] else None for i in range(L)]

    event_type = np.where(is_trade, "trade", "quote_update").astype(object)

    c = condition.detach().cpu().numpy().astype(np.int64)
    if c.shape != (len(CONDITION_COLUMNS),):
        raise ValueError(f"condition has shape {c.shape}, expected ({len(CONDITION_COLUMNS)},)")
    label = _regime_label(c[0], c[1])

    # mid_return for the canonical schema (for inspection / debugging downstream).
    mid_return_canonical = [None] + np.diff(np.log(mid)).tolist()

    df = pl.DataFrame({
        "event_idx":    np.arange(L, dtype=np.int64),
        "ts_ns":        start_ts_ns + np.arange(L, dtype=np.int64) * inter_event_ns,
        "ticker":       [ticker] * L,
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
        "c_vol":        np.full(L, c[0], dtype=np.int8),
        "c_vpin":       np.full(L, c[1], dtype=np.int8),
        "c_imb":        np.full(L, c[2], dtype=np.int8),
        "c_tod":        np.full(L, c[3], dtype=np.int8),
        "regime_label": [label] * L,
    })

    df = df.with_columns([
        pl.col("ticker").cast(pl.Categorical),
        pl.col("event_type").cast(pl.Categorical),
        pl.col("trade_sign").cast(pl.Int8),
        pl.col("regime_label").cast(pl.Categorical),
    ])
    # If a window has no trade events, all entries in nullable columns are
    # None and polars infers Null dtype. Re-cast to the canonical schema's
    # expected types so downstream validation passes.
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


def decode_batch_to_parquet(
    samples: torch.Tensor,           # (B, L, F)
    conditions: torch.Tensor,        # (B, n_axes)
    *,
    out_dir: str | Path,
    ticker: str,
    start_ts_ns: int,
    seed: int,
    regime_label: str,
    norm_stats: NormStats,
    tick_size_usd: float = DEFAULT_TICK_SIZE_USD,
    inter_event_ns: int = DEFAULT_INTER_EVENT_NS,
) -> list[Path]:
    """Decode a batch of synthetic windows. Each batch element → its own Parquet."""
    out_dir = Path(out_dir) / ticker / regime_label
    os.makedirs(out_dir, exist_ok=True)

    paths: list[Path] = []
    for i in range(samples.shape[0]):
        df = decode_window_to_dataframe(
            samples[i],
            condition=conditions[i],
            ticker=ticker,
            start_ts_ns=start_ts_ns,
            norm_stats=norm_stats,
            tick_size_usd=tick_size_usd,
            inter_event_ns=inter_event_ns,
        )
        schema.validate(df)
        path = out_dir / f"day_{seed:04d}_{i:03d}.parquet"
        parquet.write_tape(df, path, validate=False)
        paths.append(path)
    return paths


def decode_stitched_windows_to_dataframe(
    samples: torch.Tensor,           # (N_windows, L, F)
    *,
    condition: torch.Tensor,         # (n_axes,) — same regime for all windows
    ticker: str,
    start_ts_ns: int,
    norm_stats: NormStats,
    tick_size_usd: float = DEFAULT_TICK_SIZE_USD,
    inter_event_ns: int = DEFAULT_INTER_EVENT_NS,
) -> pl.DataFrame:
    """Stitch N consecutive sampled windows into one longer canonical tape.

    The N windows share the same regime condition. mid trajectories are
    integrated forward across windows: window k+1's anchor_mid is the END
    mid of window k, so the overall mid path is continuous. Per-window
    mid_return[0] is left as the convention from `decode_window_to_dataframe`
    (effectively 0 since we set the running anchor) — at most a tiny boundary
    artifact at window seams; documented in Work4 §1.1.

    Output: one DataFrame of N*L events on the canonical schema with
    monotonic `event_idx` and continuous `ts_ns`.
    """
    if samples.dim() != 3:
        raise ValueError(f"expected (N, L, F), got shape {tuple(samples.shape)}")
    from dataclasses import replace

    N, L, _ = samples.shape
    dfs: list[pl.DataFrame] = []
    current_anchor = float(norm_stats.anchor_mid)
    for k in range(N):
        norm_for_window = replace(norm_stats, anchor_mid=current_anchor)
        df = decode_window_to_dataframe(
            samples[k],
            condition=condition,
            ticker=ticker,
            start_ts_ns=start_ts_ns + k * L * inter_event_ns,
            norm_stats=norm_for_window,
            tick_size_usd=tick_size_usd,
            inter_event_ns=inter_event_ns,
        )
        dfs.append(df)
        current_anchor = float(df["mid"][-1])

    stitched = pl.concat(dfs, how="vertical_relaxed")
    # Re-index event_idx monotonically across the stitched tape; ts_ns is
    # already monotonic by construction (per-window offset above).
    stitched = stitched.drop("event_idx").with_row_index("event_idx").with_columns(
        pl.col("event_idx").cast(pl.Int64)
    )
    # If every concatenated window had only nulls in a nullable column (e.g.
    # no trade events at all in any window), polars collapses that column's
    # dtype to Null. Re-cast nullable float/int columns back to their schema
    # types so canonical-schema validation passes.
    nullable_recasts = {
        "trade_px": pl.Float64, "trade_sz": pl.Int64,
        "trade_sign": pl.Int8, "is_lit": pl.Boolean,
        "trade_dist": pl.Float64, "mid_return": pl.Float64,
    }
    cast_exprs = []
    for col, dtype in nullable_recasts.items():
        if col in stitched.columns and stitched.schema[col] != dtype:
            cast_exprs.append(pl.col(col).cast(dtype))
    if cast_exprs:
        stitched = stitched.with_columns(cast_exprs)
    return stitched


def decode_stitched_batch_to_parquet(
    samples: torch.Tensor,           # (N_windows, B, L, F)
    conditions: torch.Tensor,        # (B, n_axes) — same condition for all N windows of a batch element
    *,
    out_dir: str | Path,
    ticker: str,
    start_ts_ns: int,
    seed: int,
    regime_label: str,
    norm_stats: NormStats,
    tick_size_usd: float = DEFAULT_TICK_SIZE_USD,
    inter_event_ns: int = DEFAULT_INTER_EVENT_NS,
) -> list[Path]:
    """Decode B parallel stitched chains. Each batch element → one stitched Parquet
    of length N_windows × L."""
    if samples.dim() != 4:
        raise ValueError(f"expected (N, B, L, F), got shape {tuple(samples.shape)}")
    out_dir = Path(out_dir) / ticker / regime_label
    os.makedirs(out_dir, exist_ok=True)

    N, B, L, _ = samples.shape
    paths: list[Path] = []
    for b in range(B):
        # All N windows for this batch element. samples[:, b] has shape (N, L, F).
        df = decode_stitched_windows_to_dataframe(
            samples[:, b],
            condition=conditions[b],
            ticker=ticker,
            start_ts_ns=start_ts_ns,
            norm_stats=norm_stats,
            tick_size_usd=tick_size_usd,
            inter_event_ns=inter_event_ns,
        )
        schema.validate(df)
        path = out_dir / f"day_{seed:04d}_{b:03d}.parquet"
        parquet.write_tape(df, path, validate=False)
        paths.append(path)
    return paths
