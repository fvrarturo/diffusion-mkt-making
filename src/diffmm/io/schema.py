"""Canonical event-tape schema.

Real (cleaned TAQ) and synthetic (decoded from generator) Parquet files conform
to this schema. The schema is the contract that lets the fill simulator and the
agents stay blind to provenance — see Work0.md §3.
"""
from __future__ import annotations

import polars as pl

SCHEMA_VERSION = "v0.1"

EVENT_TYPES = ("quote_update", "trade")
REGIME_LABELS = ("base", "high_vol", "toxic", "thin")

# Source-data origin (TAQ specifics, recorded in Work0.md / next-step decisions):
#   bid_px / ask_px:       BID, ASK from order_*.csv.gz (after we roll up per-exchange BBO to NBBO)
#   bid_sz / ask_sz:       BIDSIZ, ASKSIZ
#   trade_px / trade_sz:   PRICE, SIZE from trade_*.csv.gz
#   trade_sign:            Lee–Ready or BVC, computed at clean-time
#   is_lit:                False iff EX == 'D' (FINRA TRF)
CORE_DTYPES: dict[str, pl.DataType] = {
    "event_idx":    pl.Int64,
    "ts_ns":        pl.Int64,         # nanoseconds since epoch (UTC); TAQ TIME_M is ET — convert at clean time
    "ticker":       pl.Categorical,
    "event_type":   pl.Categorical,   # one of EVENT_TYPES
    # Absolute prices/sizes — kept for the fill simulator (Phase 3) which
    # needs USD prices for PnL accounting. The generator does NOT see these
    # directly — see the relative features below.
    "bid_px":       pl.Float64,       # dollars
    "ask_px":       pl.Float64,
    "bid_sz":       pl.Int64,         # shares
    "ask_sz":       pl.Int64,
    "trade_px":     pl.Float64,       # nullable: only for trade events
    "trade_sz":     pl.Int64,
    "trade_sign":   pl.Int8,          # nullable: -1 / +1
    "is_lit":       pl.Boolean,       # nullable: True for lit-exchange trades, False for FINRA TRF
    "mid":          pl.Float64,       # derived: (bid_px + ask_px) / 2
    "spread":       pl.Float64,       # derived: ask_px - bid_px
    # Price-invariant relative features — what the generator actually trains
    # on. See dataset.FEATURE_COLUMNS. Replacing absolute prices with these
    # eliminates the calendar-time confound documented in Work2 §4.
    "bid_dist":     pl.Float64,       # bid_px - mid (≤ 0)
    "ask_dist":     pl.Float64,       # ask_px - mid (≥ 0)
    "trade_dist":   pl.Float64,       # nullable: trade_px - mid for trade events
    "mid_return":   pl.Float64,       # nullable: log(mid_t / mid_{t-1}); null on first row
    # Regime conditioning — set by Stage A2 finalize.
    "c_vol":        pl.Int8,          # 0/1/2: σ bucket
    "c_vpin":       pl.Int8,          # 0/1/2
    "c_imb":        pl.Int8,          # 0/1/2
    "c_tod":        pl.Int8,          # 0/1/2: morning/midday/afternoon
    "regime_label": pl.Categorical,   # one of REGIME_LABELS
}

NULLABLE = {"trade_px", "trade_sz", "trade_sign", "is_lit", "trade_dist", "mid_return"}
NON_NULLABLE = set(CORE_DTYPES) - NULLABLE


class SchemaError(ValueError):
    pass


def validate(df: pl.DataFrame, *, allow_extra: bool = True) -> pl.DataFrame:
    """Assert df conforms to the canonical schema. Returns df unchanged.

    Raises SchemaError on any violation. Set allow_extra=False to reject
    additional columns (e.g., depth columns) that are usually permitted.
    """
    missing = [c for c in CORE_DTYPES if c not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {missing}")

    if not allow_extra:
        extra = [c for c in df.columns if c not in CORE_DTYPES]
        if extra:
            raise SchemaError(f"unexpected columns: {extra}")

    for col, expected in CORE_DTYPES.items():
        actual = df.schema[col]
        if actual != expected:
            raise SchemaError(f"{col}: expected dtype {expected}, got {actual}")

    for col in NON_NULLABLE:
        if df[col].null_count() > 0:
            raise SchemaError(f"{col} has nulls but is non-nullable")

    bad = df.filter(pl.col("ask_px") <= pl.col("bid_px")).height
    if bad > 0:
        raise SchemaError(f"{bad} rows have crossed/locked book (ask_px <= bid_px)")

    if not df["event_idx"].is_sorted():
        raise SchemaError("event_idx is not sorted")

    return df


def empty_frame() -> pl.DataFrame:
    """An empty DataFrame with the canonical schema — useful as a template."""
    return pl.DataFrame(schema=CORE_DTYPES)


# ─── Intermediate (Stage A1) schema ────────────────────────────────────
# Output of per-date cleaning, BEFORE training-set quantiles are known.
# c_imb and c_tod are deterministic (fixed thresholds + clock), so they're
# already filled in. c_vol and c_vpin require quantiles fitted on training
# tapes only → defer to Stage A2 (finalize). raw_vol and raw_vpin carry the
# continuous values forward.
INTERMEDIATE_DTYPES: dict[str, pl.DataType] = {
    "event_idx":    pl.Int64,
    "ts_ns":        pl.Int64,
    "ticker":       pl.Categorical,
    "event_type":   pl.Categorical,
    "bid_px":       pl.Float64,
    "ask_px":       pl.Float64,
    "bid_sz":       pl.Int64,
    "ask_sz":       pl.Int64,
    "trade_px":     pl.Float64,
    "trade_sz":     pl.Int64,
    "trade_sign":   pl.Int8,
    "is_lit":       pl.Boolean,
    "mid":          pl.Float64,
    "spread":       pl.Float64,
    "c_imb":        pl.Int8,
    "c_tod":        pl.Int8,
    "raw_vol":      pl.Float64,
    "raw_vpin":     pl.Float64,
}

INTERMEDIATE_NULLABLE = {"trade_px", "trade_sz", "trade_sign", "is_lit", "raw_vpin"}


def validate_intermediate(df: pl.DataFrame) -> pl.DataFrame:
    """Same checks as `validate`, applied against the intermediate schema."""
    missing = [c for c in INTERMEDIATE_DTYPES if c not in df.columns]
    if missing:
        raise SchemaError(f"intermediate: missing columns {missing}")
    for col, expected in INTERMEDIATE_DTYPES.items():
        actual = df.schema[col]
        if actual != expected:
            raise SchemaError(f"intermediate: {col}: expected {expected}, got {actual}")
    for col in set(INTERMEDIATE_DTYPES) - INTERMEDIATE_NULLABLE:
        if df[col].null_count() > 0:
            raise SchemaError(f"intermediate: {col} non-nullable but has nulls")
    if not df["event_idx"].is_sorted():
        raise SchemaError("intermediate: event_idx not sorted")
    bad = df.filter(pl.col("ask_px") <= pl.col("bid_px")).height
    if bad > 0:
        raise SchemaError(f"intermediate: {bad} crossed/locked rows")
    return df


def add_relative_features(df: pl.DataFrame) -> pl.DataFrame:
    """Add price-invariant features used by the generator: bid_dist, ask_dist,
    trade_dist, mid_return.

    Operates on a frame already containing absolute bid_px/ask_px/mid/trade_px.
    Safe to call on intermediate or canonical-shaped frames; missing or
    pre-existing columns are handled idempotently.
    """
    out = df.with_columns([
        (pl.col("bid_px") - pl.col("mid")).alias("bid_dist"),
        (pl.col("ask_px") - pl.col("mid")).alias("ask_dist"),
        (pl.col("trade_px") - pl.col("mid")).alias("trade_dist"),
        # log-return of mid since the previous event; null on the first row.
        (pl.col("mid").log().diff()).alias("mid_return"),
    ])
    return out


def intermediate_to_canonical(
    df: pl.DataFrame,
    *,
    c_vol: pl.Series,
    c_vpin: pl.Series,
    regime_label: pl.Series,
) -> pl.DataFrame:
    """Drop raw_* columns and fill in regime columns to produce canonical schema."""
    out = df.drop(["raw_vol", "raw_vpin"]).with_columns([
        c_vol.alias("c_vol").cast(pl.Int8),
        c_vpin.alias("c_vpin").cast(pl.Int8),
        regime_label.alias("regime_label").cast(pl.Categorical),
    ])
    out = add_relative_features(out)
    return out.select(list(CORE_DTYPES.keys()))
