"""End-to-end Phase 1 cleaning smoke test on tiny synthetic TAQ data.

Generates a few hundred rows of TAQ-like quote and trade CSV, runs the full
pipeline (split → clean → NBBO → sign → features → quantiles → finalize), and
asserts the canonical schema validates.

Catches regressions in:
  - timestamp parsing (variable-width hours)
  - NBBO state machine
  - Lee–Ready signing
  - VPIN bucketing on small inputs
  - intermediate→canonical column conversion
  - holdout chmod
"""
from __future__ import annotations

import gzip
import os
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from diffmm.data import clean as clean_mod
from diffmm.data import features as feat_mod
from diffmm.data import raw_split, regime, splits
from diffmm.data.tapes import build_intermediate_tape
from diffmm.io import schema


# ─── synthetic-data factory ────────────────────────────────────────────

def _make_taq_quote_csv(out_path: Path, dates: list[str], rows_per_date: int = 200) -> None:
    """Write a TAQ-shaped NBBO CSV with multiple dates of synthetic quotes."""
    header = ("DATE,TIME_M,EX,BID,BIDSIZ,ASK,ASKSIZ,QU_COND,QU_SEQNUM,"
              "NATBBO_IND,QU_CANCEL,QU_SOURCE,SYM_ROOT,SYM_SUFFIX")
    rng = np.random.default_rng(0)
    lines = [header]
    for date in dates:
        # synthesize a couple of exchanges with slightly different bids/asks
        for i in range(rows_per_date):
            ex = "P" if i % 2 == 0 else "K"
            # session: 10:00 onward with ms-spaced events
            hh = 10
            mm = (i // 60) % 60
            ss = i % 60
            ns = i * 1_000_000
            time_m = f"{hh}:{mm:02d}:{ss:02d}.{ns:09d}"
            mid = 50.0 + 0.01 * (i % 100)
            spread = 0.02
            bid = round(mid - spread / 2, 4)
            ask = round(mid + spread / 2, 4)
            bid_sz = int(rng.integers(100, 500))
            ask_sz = int(rng.integers(100, 500))
            lines.append(
                f"{date},{time_m},{ex},{bid},{bid_sz},{ask},{ask_sz},R,{i},2,,N,INTC,"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("\n".join(lines) + "\n")


def _make_taq_trade_csv(out_path: Path, dates: list[str], rows_per_date: int = 80) -> None:
    header = ("DATE,TIME_M,EX,SYM_ROOT,SYM_SUFFIX,TR_SCOND,SIZE,PRICE,TR_STOP_IND,"
              "TR_CORR,TR_SEQNUM,TR_ID,TR_SOURCE,TR_RF")
    rng = np.random.default_rng(1)
    lines = [header]
    for date in dates:
        for i in range(rows_per_date):
            ex = "P"
            hh = 10
            mm = (i // 8) % 60
            ss = (i * 7) % 60
            ns = i * 5_000_000
            time_m = f"{hh}:{mm:02d}:{ss:02d}.{ns:09d}"
            price = round(50.0 + 0.01 * (i % 100) + rng.standard_normal() * 0.005, 4)
            size = int(rng.integers(1, 100))
            scond = "@ TI"
            lines.append(
                f"{date},{time_m},{ex},INTC,,{scond},{size},{price},,00,{i},{i},N,"
            )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write("\n".join(lines) + "\n")


# ─── tests ─────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_pipeline(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    raw_part = tmp_path / "raw_partitioned"
    inter = tmp_path / "intermediate"
    out = tmp_path / "processed"
    dates = ["2025-04-02", "2025-04-03", "2025-04-04"]
    quotes_gz = raw_dir / "order_INTC.csv.gz"
    trades_gz = raw_dir / "trade_INTC.csv.gz"
    _make_taq_quote_csv(quotes_gz, dates, rows_per_date=400)
    _make_taq_trade_csv(trades_gz, dates, rows_per_date=120)
    yield {
        "dates": dates,
        "quotes_gz": quotes_gz, "trades_gz": trades_gz,
        "raw_part": raw_part, "inter": inter, "out": out,
    }


def test_split_gzip_by_date_writes_one_parquet_per_date(tmp_pipeline):
    out_q = tmp_pipeline["raw_part"] / "INTC" / "quotes"
    out_t = tmp_pipeline["raw_part"] / "INTC" / "trades"
    written_q = raw_split.split_gzip_by_date(tmp_pipeline["quotes_gz"], out_q, kind="quotes")
    written_t = raw_split.split_gzip_by_date(tmp_pipeline["trades_gz"], out_t, kind="trades")
    assert len(written_q) == len(tmp_pipeline["dates"])
    assert len(written_t) == len(tmp_pipeline["dates"])
    # files are named {date}.parquet
    assert {p.stem for p in written_q} == set(tmp_pipeline["dates"])


def test_intermediate_tape_validates_against_intermediate_schema(tmp_pipeline):
    out_q = tmp_pipeline["raw_part"] / "INTC" / "quotes"
    out_t = tmp_pipeline["raw_part"] / "INTC" / "trades"
    raw_split.split_gzip_by_date(tmp_pipeline["quotes_gz"], out_q, kind="quotes")
    raw_split.split_gzip_by_date(tmp_pipeline["trades_gz"], out_t, kind="trades")

    date = tmp_pipeline["dates"][0]
    df = build_intermediate_tape(
        ticker="INTC", date=date,
        raw_quotes_parquet=out_q / f"{date}.parquet",
        raw_trades_parquet=out_t / f"{date}.parquet",
        clean_cfg=clean_mod.CleanConfig(),
        feat_cfg=feat_mod.FeatureConfig(vol_window_seconds=60, vpin_n_buckets_per_day=5,
                                        vpin_window_buckets=5),
    )
    schema.validate_intermediate(df)
    assert df.height > 0
    # both event types must appear
    et = set(df["event_type"].drop_nulls().unique().to_list())
    assert "quote_update" in et
    assert "trade" in et


def test_finalize_makes_canonical_tapes_and_locks_holdout(tmp_pipeline):
    # Stage A0
    out_q = tmp_pipeline["raw_part"] / "INTC" / "quotes"
    out_t = tmp_pipeline["raw_part"] / "INTC" / "trades"
    raw_split.split_gzip_by_date(tmp_pipeline["quotes_gz"], out_q, kind="quotes")
    raw_split.split_gzip_by_date(tmp_pipeline["trades_gz"], out_t, kind="trades")

    # Stage A1
    inter_dir = tmp_pipeline["inter"]
    for date in tmp_pipeline["dates"]:
        df = build_intermediate_tape(
            ticker="INTC", date=date,
            raw_quotes_parquet=out_q / f"{date}.parquet",
            raw_trades_parquet=out_t / f"{date}.parquet",
            clean_cfg=clean_mod.CleanConfig(),
            feat_cfg=feat_mod.FeatureConfig(vol_window_seconds=60, vpin_n_buckets_per_day=5,
                                            vpin_window_buckets=5),
        )
        path = inter_dir / "INTC" / f"{date}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        df.write_parquet(path)

    # Stage A2 — splits with tiny sizes (1/1/1) so all 3 dates fit
    sp = splits.make_splits(
        tmp_pipeline["dates"],
        splits.SplitConfig(n_train_days=1, n_val_days=1, n_holdout_days=1, seed=0),
    )
    train_paths = [inter_dir / "INTC" / f"{d}.parquet" for d in sp.train]
    q = regime.fit_quantiles(train_paths)

    out_dir = tmp_pipeline["out"]
    written = splits.materialize_split(
        ticker="INTC", splits=sp,
        intermediate_dir=inter_dir, output_dir=out_dir, quantiles=q,
    )
    assert sum(len(v) for v in written.values()) == 3

    # Holdout file is read-only.
    holdout_file = written["holdout"][0]
    assert (os.stat(holdout_file).st_mode & 0o222) == 0, "holdout should be read-only"

    # Canonical schema validates.
    train_file = written["train"][0]
    df = pl.read_parquet(train_file)
    schema.validate(df)

    # Restore writability so pytest can clean up tmp dirs.
    os.chmod(holdout_file, 0o644)


def test_apply_quantiles_label_consistency():
    """Every (c_vol, c_vpin) combination must map to the documented label —
    catches the categorical-join drift bug that leaked in the prior version.
    """
    # Build a tiny intermediate frame with multiple rows per (c_vol, c_vpin) cell so
    # that mid_return (computed via diff at canonical-time) has non-null entries.
    rows = []
    for cv in (0, 1, 2):
        for cp in (0, 1, 2):
            for k in range(3):  # 3 events per cell
                rows.append(dict(
                    event_idx=len(rows),
                    ts_ns=1_700_000_000_000_000_000 + len(rows),
                    ticker="INTC",
                    event_type="quote_update",
                    bid_px=10.0 + 0.001 * k, ask_px=10.02 + 0.001 * k,
                    bid_sz=100, ask_sz=100,
                    trade_px=None, trade_sz=None, trade_sign=None, is_lit=None,
                    mid=10.01 + 0.001 * k, spread=0.02,
                    c_imb=1, c_tod=1,
                    raw_vol={0: 0.0, 1: 0.5, 2: 1.0}[cv],
                    raw_vpin={0: 0.0, 1: 0.5, 2: 1.0}[cp],
                ))
    df = pl.DataFrame(rows, schema_overrides=schema.INTERMEDIATE_DTYPES)
    schema.validate_intermediate(df)

    q = regime.Quantiles(vol=(0.25, 0.75), vpin=(0.25, 0.75))
    out = regime.apply_quantiles(df, q)

    expected = {
        (0, 0): "base", (0, 1): "base", (0, 2): "thin",
        (1, 0): "base", (1, 1): "base", (1, 2): "thin",
        (2, 0): "high_vol", (2, 1): "high_vol", (2, 2): "toxic",
    }
    for (cv, cp), label in expected.items():
        actual = out.filter((pl.col("c_vol") == cv) & (pl.col("c_vpin") == cp))["regime_label"][0]
        assert actual == label, f"({cv},{cp}) → expected {label}, got {actual}"


def test_absolute_relative_round_trip():
    """absolute prices → relative features → decode → absolute prices must
    recover the originals to float32 epsilon. This is the correctness gate
    for the price-invariant representation added in Work3.
    """
    import numpy as np
    import torch
    from diffmm.data.dataset import FEATURE_COLUMNS, NormStats, N_FEATURES
    from diffmm.generator.decode import decode_window_to_dataframe
    from diffmm.io import schema as schema_mod

    # Synthesize a plausible 32-event INTC window in ABSOLUTE-price space.
    rng = np.random.default_rng(0)
    L = 32
    base_mid = 37.50
    # log-returns ~ N(0, 1e-5) — realistic event-scale INTC volatility
    log_returns = rng.normal(0.0, 1e-5, size=L)
    log_returns[0] = 0.0
    mid = base_mid * np.exp(np.cumsum(log_returns))
    half_spread = np.full(L, 0.005)  # 1-tick spread
    bid_px = mid - half_spread
    ask_px = mid + half_spread
    bid_sz = rng.integers(100, 500, size=L)
    ask_sz = rng.integers(100, 500, size=L)
    # A few trades scattered through the window
    is_trade = rng.random(L) < 0.3
    trade_px = np.where(is_trade, mid + rng.normal(0.0, 0.002, size=L), np.nan)
    trade_sz = np.where(is_trade, rng.integers(1, 100, size=L), 0)

    # Compute the relative features (what the generator would produce).
    bid_dist = bid_px - mid
    ask_dist = ask_px - mid
    trade_dist = np.where(is_trade, trade_px - mid, 0.0)
    mid_return = np.concatenate([[0.0], np.diff(np.log(mid))])

    # Pack into a (L, 8) tensor in FEATURE_COLUMNS order.
    feats = np.stack([
        bid_dist, ask_dist, bid_sz.astype(np.float64), ask_sz.astype(np.float64),
        trade_dist, trade_sz.astype(np.float64),
        np.where(is_trade, np.sign(trade_dist), 0.0),
        mid_return,
    ], axis=1).astype(np.float32)
    assert feats.shape == (L, N_FEATURES)

    # Identity norm stats so the test isolates the relative→absolute math.
    norm = NormStats(
        mean=np.zeros(N_FEATURES, dtype=np.float32),
        std=np.ones(N_FEATURES, dtype=np.float32),
        feature_columns=FEATURE_COLUMNS,
        anchor_mid=float(base_mid),
    )
    cond = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    df = decode_window_to_dataframe(
        torch.from_numpy(feats),
        condition=cond,
        ticker="INTC",
        start_ts_ns=1_700_000_000_000_000_000,
        norm_stats=norm,
    )
    schema_mod.validate(df)
    assert df.height == L

    # Compare recovered absolute prices to the originals.
    tol = 5e-5  # float32 + cumulative log/exp round-off over 32 events
    recovered_mid = df["mid"].to_numpy()
    assert np.allclose(recovered_mid, mid, atol=tol), (
        f"mid drift beyond {tol}: max abs err {np.abs(recovered_mid - mid).max()}"
    )
    recovered_bid = df["bid_px"].to_numpy()
    recovered_ask = df["ask_px"].to_numpy()
    assert np.allclose(recovered_bid, bid_px, atol=tol)
    assert np.allclose(recovered_ask, ask_px, atol=tol)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
