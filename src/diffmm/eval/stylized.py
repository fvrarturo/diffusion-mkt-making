"""Stylized-fact validation suite for the diffusion generator.

Implements Parts B (marginal distributions), C (temporal dependence), and D
(cross-feature / conditional structure) of the Work5 validation plan.

All functions take pre-concatenated Polars DataFrames on the canonical schema
and return a result dataclass with the numerical findings. Plotting lives in
`diffmm.viz.validation`.

Convention:
  - "real" pool   = concatenated real validation tapes
  - "synth" pool  = concatenated synthetic tapes (subsample if very large)
  - All comparisons are real vs synth; training data only used for norm stats
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats as scistats
from scipy.spatial.distance import jensenshannon


# ─── helpers ───────────────────────────────────────────────────────────

def _ensure_numpy(s: pl.Series | np.ndarray) -> np.ndarray:
    if isinstance(s, np.ndarray):
        return s
    return s.to_numpy()


def _drop_non_finite(x: np.ndarray) -> np.ndarray:
    return x[np.isfinite(x)]


def _summary_stats(x: np.ndarray) -> dict:
    """Summary statistics dict, suitable for JSON / CSV export."""
    if x.size == 0:
        return {k: float("nan") for k in
                ("mean", "std", "skew", "kurt", "min", "max", "p1", "p5", "p95", "p99", "n")}
    return {
        "mean": float(np.mean(x)),
        "std":  float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        "skew": float(scistats.skew(x)),
        "kurt": float(scistats.kurtosis(x, fisher=True)),  # excess kurt
        "min":  float(np.min(x)),
        "max":  float(np.max(x)),
        "p1":   float(np.percentile(x, 1)),
        "p5":   float(np.percentile(x, 5)),
        "p95":  float(np.percentile(x, 95)),
        "p99":  float(np.percentile(x, 99)),
        "n":    int(x.size),
    }


def inter_day_wasserstein(per_day: list[np.ndarray]) -> tuple[float, float]:
    """Mean ± std of W-1 between every pair of per-day samples."""
    pairs = list(itertools.combinations(range(len(per_day)), 2))
    if not pairs:
        return 0.0, 0.0
    ws = np.array([
        scistats.wasserstein_distance(per_day[i], per_day[j])
        for i, j in pairs
    ])
    return float(ws.mean()), float(ws.std(ddof=1) if ws.size > 1 else 0.0)


# ─── Part B — Marginal Distribution Tests ─────────────────────────────

@dataclass
class ReturnStats:
    real: dict
    synth: dict
    wasserstein_1: float
    inter_day_w1_mean: float
    inter_day_w1_std: float
    ks_statistic: float
    ks_pvalue: float
    pass_w1: bool                            # W-1 < 2× inter-day mean
    pass_kurtosis: bool                      # synth kurt > 0
    pass_skewness: bool                      # |synth skew - real skew| < 0.5
    real_values: np.ndarray = field(repr=False)
    synth_values: np.ndarray = field(repr=False)


def return_diagnostics(
    real_per_day: list[pl.DataFrame],
    synth_concat: pl.DataFrame,
) -> ReturnStats:
    """B.1 — mid-return marginal comparison."""
    real_arrays = [_drop_non_finite(_ensure_numpy(d["mid_return"].drop_nulls()))
                   for d in real_per_day]
    real_arrays = [a for a in real_arrays if a.size > 0]
    real_all = np.concatenate(real_arrays) if real_arrays else np.zeros(0)
    synth_all = _drop_non_finite(_ensure_numpy(
        synth_concat["mid_return"].drop_nulls()
    ))

    real_summary = _summary_stats(real_all)
    synth_summary = _summary_stats(synth_all)
    w1 = scistats.wasserstein_distance(real_all, synth_all) if real_all.size and synth_all.size else float("nan")
    inter_w1_mean, inter_w1_std = inter_day_wasserstein(real_arrays)

    if real_all.size and synth_all.size:
        ks = scistats.ks_2samp(real_all, synth_all)
        ks_stat, ks_p = float(ks.statistic), float(ks.pvalue)
    else:
        ks_stat, ks_p = float("nan"), float("nan")

    return ReturnStats(
        real=real_summary,
        synth=synth_summary,
        wasserstein_1=float(w1),
        inter_day_w1_mean=inter_w1_mean,
        inter_day_w1_std=inter_w1_std,
        ks_statistic=ks_stat,
        ks_pvalue=ks_p,
        pass_w1=(w1 < 2.0 * inter_w1_mean) if inter_w1_mean > 0 else False,
        pass_kurtosis=(synth_summary["kurt"] > 0),
        pass_skewness=(abs(synth_summary["skew"] - real_summary["skew"]) < 0.5),
        real_values=real_all,
        synth_values=synth_all,
    )


@dataclass
class SpreadStats:
    real_one_tick_frac: float
    synth_one_tick_frac: float
    real_pmf: dict[str, float]              # bin label → fraction
    synth_pmf: dict[str, float]
    wasserstein_1: float
    inter_day_w1_mean: float
    chi_sq_statistic: float
    chi_sq_pvalue: float
    pass_one_tick: bool                     # synth >80%
    pass_w1: bool
    real_values: np.ndarray = field(repr=False)
    synth_values: np.ndarray = field(repr=False)


def _spread_pmf(spreads: np.ndarray, tick: float = 0.01) -> dict[str, float]:
    """Discrete PMF over {1¢, 2¢, 3¢, 4¢, ≥5¢}."""
    bins = np.array([0.5 * tick, 1.5 * tick, 2.5 * tick, 3.5 * tick, 4.5 * tick, np.inf])
    labels = ["1c", "2c", "3c", "4c", ">=5c"]
    if spreads.size == 0:
        return {lbl: 0.0 for lbl in labels}
    counts, _ = np.histogram(spreads, bins=bins)
    fracs = counts / counts.sum()
    return dict(zip(labels, [float(f) for f in fracs]))


def spread_diagnostics(
    real_per_day: list[pl.DataFrame],
    synth_concat: pl.DataFrame,
    tick_size_usd: float = 0.01,
) -> SpreadStats:
    """B.2 — spread distribution + 1-tick fraction + chi-squared on PMF."""
    real_arrays = [_ensure_numpy(d["spread"]) for d in real_per_day]
    real_all = np.concatenate(real_arrays) if real_arrays else np.zeros(0)
    synth_all = _ensure_numpy(synth_concat["spread"])

    real_one_tick = float(np.mean(np.abs(real_all - tick_size_usd) < 1e-3)) if real_all.size else 0.0
    synth_one_tick = float(np.mean(np.abs(synth_all - tick_size_usd) < 1e-3)) if synth_all.size else 0.0

    real_pmf = _spread_pmf(real_all, tick_size_usd)
    synth_pmf = _spread_pmf(synth_all, tick_size_usd)
    w1 = scistats.wasserstein_distance(real_all, synth_all) if real_all.size and synth_all.size else float("nan")
    inter_w1_mean, _ = inter_day_wasserstein(real_arrays)

    # Chi-squared on the PMF buckets — apply small-sample protection
    obs = np.array(list(synth_pmf.values())) * (synth_all.size or 1)
    exp = np.array(list(real_pmf.values())) * (synth_all.size or 1)
    if exp.min() > 5 and synth_all.size > 0:
        # Standard chi-square requires expected counts > 5
        ch = scistats.chisquare(obs, exp)
        ch_stat, ch_p = float(ch.statistic), float(ch.pvalue)
    else:
        ch_stat, ch_p = float("nan"), float("nan")

    return SpreadStats(
        real_one_tick_frac=real_one_tick,
        synth_one_tick_frac=synth_one_tick,
        real_pmf=real_pmf,
        synth_pmf=synth_pmf,
        wasserstein_1=float(w1),
        inter_day_w1_mean=inter_w1_mean,
        chi_sq_statistic=ch_stat,
        chi_sq_pvalue=ch_p,
        pass_one_tick=(synth_one_tick > 0.80),
        pass_w1=(w1 < 2.0 * inter_w1_mean) if inter_w1_mean > 0 else False,
        real_values=real_all,
        synth_values=synth_all,
    )


@dataclass
class SizeStats:
    real_summary: dict
    synth_summary: dict
    log_w1: float                            # W-1 on log(size+1)
    inter_day_log_w1_mean: float
    tail_alpha_real: float
    tail_alpha_synth: float
    pass_tail: bool                          # |Δα| < 30%
    pass_w1: bool
    real_log_values: np.ndarray = field(repr=False)
    synth_log_values: np.ndarray = field(repr=False)


def _power_law_alpha(x: np.ndarray) -> float:
    """Crude power-law tail exponent via log-log regression on the upper half."""
    x = x[x > 0]
    if x.size < 100:
        return float("nan")
    median = np.median(x)
    tail = np.sort(x[x > median])
    if tail.size < 50:
        return float("nan")
    # Empirical CCDF: P(X > x_i) = (n - i) / n
    rank = np.arange(1, tail.size + 1)
    ccdf = (tail.size - rank + 1) / tail.size
    log_x = np.log(tail)
    log_p = np.log(ccdf + 1e-12)
    # Fit ccdf ~ x^{-α} → log(ccdf) = -α log(x) + const
    slope, _ = np.polyfit(log_x, log_p, 1)
    return float(-slope)


def size_diagnostics(
    real_per_day: list[pl.DataFrame],
    synth_concat: pl.DataFrame,
) -> SizeStats:
    """B.3 — bid_sz distribution on log scale + power-law tail."""
    real_arrays = [_ensure_numpy(d["bid_sz"]).astype(np.float64) for d in real_per_day]
    real_all = np.concatenate(real_arrays) if real_arrays else np.zeros(0)
    synth_all = _ensure_numpy(synth_concat["bid_sz"]).astype(np.float64)

    real_log = np.log(np.maximum(real_all, 0) + 1)
    synth_log = np.log(np.maximum(synth_all, 0) + 1)
    inter_per_day_log = [np.log(np.maximum(a, 0) + 1) for a in real_arrays]

    w1_log = scistats.wasserstein_distance(real_log, synth_log) if real_log.size and synth_log.size else float("nan")
    inter_log_mean, _ = inter_day_wasserstein(inter_per_day_log)

    a_real = _power_law_alpha(real_all)
    a_synth = _power_law_alpha(synth_all)

    return SizeStats(
        real_summary=_summary_stats(real_all),
        synth_summary=_summary_stats(synth_all),
        log_w1=float(w1_log),
        inter_day_log_w1_mean=inter_log_mean,
        tail_alpha_real=a_real,
        tail_alpha_synth=a_synth,
        pass_tail=(not np.isnan(a_real) and not np.isnan(a_synth)
                   and abs(a_synth - a_real) / max(abs(a_real), 1e-9) < 0.3),
        pass_w1=(w1_log < 2.0 * inter_log_mean) if inter_log_mean > 0 else False,
        real_log_values=real_log,
        synth_log_values=synth_log,
    )


@dataclass
class EventCompositionStats:
    real_trade_frac: float
    synth_trade_frac: float
    synth_trade_frac_by_regime: dict[str, float]
    real_buy_frac: float
    synth_buy_frac: float
    real_inter_trade_gap: np.ndarray = field(repr=False)
    synth_inter_trade_gap: np.ndarray = field(repr=False)
    real_trade_sz_summary: dict = field(default_factory=dict)
    synth_trade_sz_summary: dict = field(default_factory=dict)


def _inter_trade_gaps(df: pl.DataFrame) -> np.ndarray:
    is_trade = (df["event_type"] == "trade").to_numpy()
    idx = np.where(is_trade)[0]
    if idx.size < 2:
        return np.zeros(0)
    return np.diff(idx)


def event_composition(
    real_concat: pl.DataFrame,
    synth_concat: pl.DataFrame,
) -> EventCompositionStats:
    """B.4 — trade fraction, trade-sign balance, inter-trade clustering."""
    real_is_trade = (real_concat["event_type"] == "trade").to_numpy()
    synth_is_trade = (synth_concat["event_type"] == "trade").to_numpy()
    real_trade_frac = float(real_is_trade.mean()) if real_is_trade.size else 0.0
    synth_trade_frac = float(synth_is_trade.mean()) if synth_is_trade.size else 0.0

    by_regime: dict[str, float] = {}
    if "regime_label" in synth_concat.columns:
        for label in ("base", "high_vol", "toxic", "thin"):
            sub = synth_concat.filter(pl.col("regime_label") == label)
            if sub.height:
                by_regime[label] = float((sub["event_type"] == "trade").mean())

    real_signs = _ensure_numpy(real_concat.filter(pl.col("event_type") == "trade")["trade_sign"].drop_nulls())
    synth_signs = _ensure_numpy(synth_concat.filter(pl.col("event_type") == "trade")["trade_sign"].drop_nulls())
    real_buy_frac = float(np.mean(real_signs > 0)) if real_signs.size else 0.5
    synth_buy_frac = float(np.mean(synth_signs > 0)) if synth_signs.size else 0.5

    real_gaps = _inter_trade_gaps(real_concat)
    synth_gaps = _inter_trade_gaps(synth_concat)

    real_trade_sz = _ensure_numpy(real_concat.filter(pl.col("event_type") == "trade")["trade_sz"].drop_nulls())
    synth_trade_sz = _ensure_numpy(synth_concat.filter(pl.col("event_type") == "trade")["trade_sz"].drop_nulls())

    return EventCompositionStats(
        real_trade_frac=real_trade_frac,
        synth_trade_frac=synth_trade_frac,
        synth_trade_frac_by_regime=by_regime,
        real_buy_frac=real_buy_frac,
        synth_buy_frac=synth_buy_frac,
        real_inter_trade_gap=real_gaps,
        synth_inter_trade_gap=synth_gaps,
        real_trade_sz_summary=_summary_stats(real_trade_sz.astype(np.float64)),
        synth_trade_sz_summary=_summary_stats(synth_trade_sz.astype(np.float64)),
    )


def marginal_summary(
    real_per_day: list[pl.DataFrame],
    synth_concat: pl.DataFrame,
) -> pd.DataFrame:
    """B.5 — per-feature marginal summary table (one row per feature)."""
    features = ("mid_return", "bid_dist", "ask_dist", "trade_dist",
                "bid_sz", "ask_sz", "trade_sz", "trade_sign", "spread")
    rows = []
    for feat in features:
        if feat not in synth_concat.columns:
            continue
        real_arrays = [_drop_non_finite(_ensure_numpy(d[feat].drop_nulls()).astype(np.float64))
                       for d in real_per_day]
        real_arrays = [a for a in real_arrays if a.size > 0]
        if not real_arrays:
            continue
        real_all = np.concatenate(real_arrays)
        synth_all = _drop_non_finite(_ensure_numpy(synth_concat[feat].drop_nulls()).astype(np.float64))
        if synth_all.size == 0:
            continue
        w1 = scistats.wasserstein_distance(real_all, synth_all)
        inter_w1_mean, _ = inter_day_wasserstein(real_arrays)
        rows.append({
            "feature": feat,
            "real_mean": float(real_all.mean()),
            "synth_mean": float(synth_all.mean()),
            "real_std": float(real_all.std(ddof=1)),
            "synth_std": float(synth_all.std(ddof=1)),
            "w1": float(w1),
            "inter_day_w1_ref": inter_w1_mean,
            "pass": bool(w1 < 2.0 * inter_w1_mean) if inter_w1_mean > 0 else False,
        })
    return pd.DataFrame(rows)


# ─── Part C — Temporal Dependence ──────────────────────────────────────

@dataclass
class ACFResult:
    name: str
    max_lag: int
    real_concat: np.ndarray                  # (max_lag+1,)
    synth_concat: np.ndarray
    real_per_day: np.ndarray                 # (n_days, max_lag+1)
    real_band_lower: np.ndarray              # (max_lag+1,)
    real_band_upper: np.ndarray
    lag_50_deviation: float                  # only if max_lag >= 50
    power_law_beta_real: float
    power_law_beta_synth: float
    pass_lag50: bool
    pass_power_law: bool


def _safe_acf(x: np.ndarray, nlags: int) -> np.ndarray:
    from statsmodels.tsa.stattools import acf
    if x.size < nlags + 2 or np.std(x) == 0:
        return np.full(nlags + 1, np.nan)
    return acf(x, nlags=nlags, fft=True)


def _power_law_acf_decay(acf_vals: np.ndarray, lo: int = 10, hi: int = 200) -> float:
    """Fit ACF ~ lag^{-β} on log-log; return β."""
    hi = min(hi, len(acf_vals) - 1)
    if hi <= lo:
        return float("nan")
    lags = np.arange(lo, hi + 1)
    vals = acf_vals[lo:hi + 1]
    mask = (vals > 0) & np.isfinite(vals)
    if mask.sum() < 5:
        return float("nan")
    slope, _ = np.polyfit(np.log(lags[mask]), np.log(vals[mask]), 1)
    return float(-slope)


def acf_abs_return(
    real_per_day: list[pl.DataFrame],
    synth_per_tape: list[pl.DataFrame],
    max_lag: int = 200,
) -> ACFResult:
    """C.1 — autocorrelation of |mid_return| (volatility clustering)."""
    real_per_day_arrays = [
        _ensure_numpy(d["mid_return"].fill_null(0).abs())
        for d in real_per_day
    ]
    real_concat = np.concatenate(real_per_day_arrays)
    synth_concat = np.concatenate([
        _ensure_numpy(d["mid_return"].fill_null(0).abs())
        for d in synth_per_tape
    ])

    real_concat_acf = _safe_acf(real_concat, max_lag)
    synth_concat_acf = _safe_acf(synth_concat, max_lag)

    per_day = np.stack([_safe_acf(a, max_lag) for a in real_per_day_arrays])
    band_low = np.nanmean(per_day, axis=0) - 2 * np.nanstd(per_day, axis=0)
    band_high = np.nanmean(per_day, axis=0) + 2 * np.nanstd(per_day, axis=0)

    if max_lag >= 50:
        r50 = real_concat_acf[50]
        s50 = synth_concat_acf[50]
        dev = abs(s50 - r50) / max(abs(r50), 1e-9) if not np.isnan(r50) else float("nan")
    else:
        dev = float("nan")

    b_real = _power_law_acf_decay(real_concat_acf)
    b_synth = _power_law_acf_decay(synth_concat_acf)

    return ACFResult(
        name="|mid_return|",
        max_lag=max_lag,
        real_concat=real_concat_acf,
        synth_concat=synth_concat_acf,
        real_per_day=per_day,
        real_band_lower=band_low,
        real_band_upper=band_high,
        lag_50_deviation=float(dev),
        power_law_beta_real=b_real,
        power_law_beta_synth=b_synth,
        pass_lag50=(dev < 0.5) if not np.isnan(dev) else False,
        pass_power_law=(not np.isnan(b_real) and not np.isnan(b_synth)
                       and abs(b_synth - b_real) / max(abs(b_real), 1e-9) < 0.5),
    )


def acf_trade_sign(
    real_per_day: list[pl.DataFrame],
    synth_per_tape: list[pl.DataFrame],
    max_lag: int = 100,
) -> ACFResult:
    """C.2 — autocorrelation of trade_sign (order-flow persistence)."""
    real_arrays = []
    for d in real_per_day:
        s = d.filter(pl.col("event_type") == "trade")["trade_sign"].drop_nulls().to_numpy().astype(np.float64)
        if s.size > max_lag + 2:
            real_arrays.append(s)
    synth_arrays = []
    for d in synth_per_tape:
        s = d.filter(pl.col("event_type") == "trade")["trade_sign"].drop_nulls().to_numpy().astype(np.float64)
        if s.size > max_lag + 2:
            synth_arrays.append(s)

    real_concat = np.concatenate(real_arrays) if real_arrays else np.zeros(0)
    synth_concat = np.concatenate(synth_arrays) if synth_arrays else np.zeros(0)
    real_concat_acf = _safe_acf(real_concat, max_lag)
    synth_concat_acf = _safe_acf(synth_concat, max_lag)

    per_day = np.stack([_safe_acf(a, max_lag) for a in real_arrays]) if real_arrays else np.zeros((0, max_lag + 1))
    if per_day.size:
        band_low = np.nanmean(per_day, axis=0) - 2 * np.nanstd(per_day, axis=0)
        band_high = np.nanmean(per_day, axis=0) + 2 * np.nanstd(per_day, axis=0)
    else:
        band_low = band_high = np.full(max_lag + 1, np.nan)

    return ACFResult(
        name="trade_sign",
        max_lag=max_lag,
        real_concat=real_concat_acf,
        synth_concat=synth_concat_acf,
        real_per_day=per_day,
        real_band_lower=band_low,
        real_band_upper=band_high,
        lag_50_deviation=float("nan"),
        power_law_beta_real=float("nan"),
        power_law_beta_synth=float("nan"),
        pass_lag50=(real_concat_acf[1] > 0 and synth_concat_acf[1] > 0
                   and abs(synth_concat_acf[1] - real_concat_acf[1]) / max(abs(real_concat_acf[1]), 1e-9) < 0.5),
        pass_power_law=False,
    )


def acf_return(
    real_per_day: list[pl.DataFrame],
    synth_per_tape: list[pl.DataFrame],
    max_lag: int = 50,
) -> ACFResult:
    """C.3 — autocorrelation of raw mid_return (bid-ask bounce)."""
    real_arrays = [_ensure_numpy(d["mid_return"].fill_null(0)) for d in real_per_day]
    synth_arrays = [_ensure_numpy(d["mid_return"].fill_null(0)) for d in synth_per_tape]
    real_concat = np.concatenate(real_arrays)
    synth_concat = np.concatenate(synth_arrays)
    real_concat_acf = _safe_acf(real_concat, max_lag)
    synth_concat_acf = _safe_acf(synth_concat, max_lag)
    per_day = np.stack([_safe_acf(a, max_lag) for a in real_arrays])
    band_low = np.nanmean(per_day, axis=0) - 2 * np.nanstd(per_day, axis=0)
    band_high = np.nanmean(per_day, axis=0) + 2 * np.nanstd(per_day, axis=0)
    return ACFResult(
        name="mid_return",
        max_lag=max_lag,
        real_concat=real_concat_acf,
        synth_concat=synth_concat_acf,
        real_per_day=per_day,
        real_band_lower=band_low,
        real_band_upper=band_high,
        lag_50_deviation=float("nan"),
        power_law_beta_real=float("nan"),
        power_law_beta_synth=float("nan"),
        pass_lag50=True,
        pass_power_law=True,
    )


@dataclass
class RVSignatureResult:
    deltas: np.ndarray
    real_mean: np.ndarray
    real_band_low: np.ndarray
    real_band_high: np.ndarray
    synth_mean: np.ndarray
    synth_band_low: np.ndarray
    synth_band_high: np.ndarray
    ratio_real: float
    ratio_synth: float
    pass_shape: bool


def _per_tape_rv(df: pl.DataFrame, deltas: list[int]) -> dict[int, float]:
    mid = df["mid"].to_numpy().astype(np.float64)
    out = {}
    for d in deltas:
        if mid.size <= d:
            out[d] = float("nan")
            continue
        sub = mid[::d]
        if sub.size < 2:
            out[d] = float("nan")
            continue
        rets = np.diff(np.log(np.maximum(sub, 1e-9)))
        out[d] = float(np.sum(rets ** 2)) / max(sub.size - 1, 1)  # per-event RV
    return out


def rv_signature(
    real_per_day: list[pl.DataFrame],
    synth_per_tape: list[pl.DataFrame],
    deltas: list[int] = (1, 2, 5, 10, 25, 50, 100, 256),
) -> RVSignatureResult:
    """C.4 — Andersen–Bollerslev–Diebold realized-variance signature."""
    deltas = list(deltas)
    real_rvs = np.array([[_per_tape_rv(d, deltas)[k] for k in deltas] for d in real_per_day])
    synth_rvs = np.array([[_per_tape_rv(d, deltas)[k] for k in deltas] for d in synth_per_tape])

    real_mean = np.nanmean(real_rvs, axis=0)
    real_std = np.nanstd(real_rvs, axis=0)
    synth_mean = np.nanmean(synth_rvs, axis=0)
    synth_std = np.nanstd(synth_rvs, axis=0)

    ratio_real = float(real_mean[0] / max(real_mean[-1], 1e-12))
    ratio_synth = float(synth_mean[0] / max(synth_mean[-1], 1e-12))
    pass_shape = (0.5 < ratio_synth / max(ratio_real, 1e-9) < 2.0)

    return RVSignatureResult(
        deltas=np.array(deltas),
        real_mean=real_mean,
        real_band_low=real_mean - 2 * real_std,
        real_band_high=real_mean + 2 * real_std,
        synth_mean=synth_mean,
        synth_band_low=synth_mean - 2 * synth_std,
        synth_band_high=synth_mean + 2 * synth_std,
        ratio_real=ratio_real,
        ratio_synth=ratio_synth,
        pass_shape=pass_shape,
    )


def _ofi_proxy(df: pl.DataFrame) -> np.ndarray:
    bid_px = df["bid_px"].to_numpy().astype(np.float64)
    ask_px = df["ask_px"].to_numpy().astype(np.float64)
    bid_sz = df["bid_sz"].to_numpy().astype(np.float64)
    ask_sz = df["ask_sz"].to_numpy().astype(np.float64)
    d_bid = np.diff(bid_sz, prepend=bid_sz[:1])
    d_ask = np.diff(ask_sz, prepend=ask_sz[:1])
    bid_up = (bid_px >= np.r_[bid_px[0], bid_px[:-1]]).astype(float)
    ask_dn = (ask_px <= np.r_[ask_px[0], ask_px[:-1]]).astype(float)
    return d_bid * bid_up - d_ask * ask_dn


def acf_ofi(
    real_per_day: list[pl.DataFrame],
    synth_per_tape: list[pl.DataFrame],
    max_lag: int = 100,
) -> ACFResult:
    """C.5 — OFI proxy autocorrelation."""
    real_arrays = [_ofi_proxy(d) for d in real_per_day]
    synth_arrays = [_ofi_proxy(d) for d in synth_per_tape]
    real_concat = np.concatenate(real_arrays)
    synth_concat = np.concatenate(synth_arrays)
    real_concat_acf = _safe_acf(real_concat, max_lag)
    synth_concat_acf = _safe_acf(synth_concat, max_lag)
    per_day = np.stack([_safe_acf(a, max_lag) for a in real_arrays])
    band_low = np.nanmean(per_day, axis=0) - 2 * np.nanstd(per_day, axis=0)
    band_high = np.nanmean(per_day, axis=0) + 2 * np.nanstd(per_day, axis=0)
    return ACFResult(
        name="OFI proxy",
        max_lag=max_lag,
        real_concat=real_concat_acf,
        synth_concat=synth_concat_acf,
        real_per_day=per_day,
        real_band_lower=band_low,
        real_band_upper=band_high,
        lag_50_deviation=float("nan"),
        power_law_beta_real=float("nan"),
        power_law_beta_synth=float("nan"),
        pass_lag50=True,
        pass_power_law=True,
    )


# ─── Part D — Cross-Feature / Conditional Structure ────────────────────

@dataclass
class CondReturnResult:
    deltas: list[int]
    bin_centers: np.ndarray
    real_means: dict[int, np.ndarray]                 # delta → (n_bins,)
    synth_means: dict[int, np.ndarray]
    real_bands: dict[int, tuple[np.ndarray, np.ndarray]]
    betas_real: dict[int, float]
    betas_synth: dict[int, float]
    r2_real: dict[int, float]
    r2_synth: dict[int, float]
    pass_sign_per_delta: dict[int, bool]


def _imbalance(df: pl.DataFrame) -> np.ndarray:
    bid_sz = df["bid_sz"].to_numpy().astype(np.float64)
    ask_sz = df["ask_sz"].to_numpy().astype(np.float64)
    s = bid_sz + ask_sz
    out = np.zeros_like(bid_sz)
    mask = s > 0
    out[mask] = (bid_sz[mask] - ask_sz[mask]) / s[mask]
    return out


def _binned_mean(x: np.ndarray, y: np.ndarray, bins: np.ndarray) -> np.ndarray:
    means = np.full(len(bins) - 1, np.nan)
    inds = np.digitize(x, bins) - 1
    for i in range(len(bins) - 1):
        sel = (inds == i)
        if sel.sum() >= 5:
            means[i] = float(np.mean(y[sel]))
    return means


def conditional_return_on_imbalance(
    real_per_day: list[pl.DataFrame],
    synth_per_tape: list[pl.DataFrame],
    n_bins: int = 20,
    deltas: list[int] = (1, 10, 50),
    n_bootstrap: int = 200,
) -> CondReturnResult:
    """D.1 — E[r_{t+Δ} | I_t]: the OFI→return signal."""
    bins = np.linspace(-1.0, 1.0, n_bins + 1)
    centers = 0.5 * (bins[1:] + bins[:-1])

    def _per_tape_imb_ret(d: pl.DataFrame, delta: int) -> tuple[np.ndarray, np.ndarray]:
        I = _imbalance(d)
        r = d["mid_return"].fill_null(0).to_numpy()
        # forward Δ-event return: r_{t+Δ} = log(mid_{t+Δ}/mid_t)
        mid = d["mid"].to_numpy().astype(np.float64)
        if mid.size <= delta:
            return I[:0], r[:0]
        fwd = np.log(np.maximum(mid[delta:], 1e-9)) - np.log(np.maximum(mid[:-delta], 1e-9))
        return I[:-delta], fwd

    real_means: dict[int, np.ndarray] = {}
    synth_means: dict[int, np.ndarray] = {}
    bands: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    betas_r: dict[int, float] = {}
    betas_s: dict[int, float] = {}
    r2_r: dict[int, float] = {}
    r2_s: dict[int, float] = {}
    pass_sign: dict[int, bool] = {}

    rng = np.random.default_rng(0)
    for delta in deltas:
        # Real concat
        I_r_all, r_r_all = [], []
        I_per_day, r_per_day = [], []
        for d in real_per_day:
            I_d, r_d = _per_tape_imb_ret(d, delta)
            if I_d.size:
                I_r_all.append(I_d); r_r_all.append(r_d)
                I_per_day.append(I_d); r_per_day.append(r_d)
        I_r = np.concatenate(I_r_all) if I_r_all else np.zeros(0)
        r_r = np.concatenate(r_r_all) if r_r_all else np.zeros(0)
        # Synth concat
        I_s_all, r_s_all = [], []
        for d in synth_per_tape:
            I_d, r_d = _per_tape_imb_ret(d, delta)
            if I_d.size:
                I_s_all.append(I_d); r_s_all.append(r_d)
        I_s = np.concatenate(I_s_all) if I_s_all else np.zeros(0)
        r_s = np.concatenate(r_s_all) if r_s_all else np.zeros(0)

        real_means[delta] = _binned_mean(I_r, r_r, bins)
        synth_means[delta] = _binned_mean(I_s, r_s, bins)

        # Bootstrap band on the real binned means by resampling days
        if len(I_per_day) >= 2:
            boot = np.empty((n_bootstrap, len(centers)))
            n = len(I_per_day)
            for b in range(n_bootstrap):
                idx = rng.integers(0, n, size=n)
                I_b = np.concatenate([I_per_day[i] for i in idx])
                r_b = np.concatenate([r_per_day[i] for i in idx])
                boot[b] = _binned_mean(I_b, r_b, bins)
            bands[delta] = (np.nanpercentile(boot, 2.5, axis=0),
                            np.nanpercentile(boot, 97.5, axis=0))
        else:
            bands[delta] = (np.full_like(centers, np.nan),
                            np.full_like(centers, np.nan))

        # Linear regression on raw event-level data
        def _lin(I_, r_):
            mask = np.isfinite(I_) & np.isfinite(r_)
            if mask.sum() < 10:
                return float("nan"), float("nan")
            slope, intercept = np.polyfit(I_[mask], r_[mask], 1)
            yhat = intercept + slope * I_[mask]
            ss_res = float(np.sum((r_[mask] - yhat) ** 2))
            ss_tot = float(np.sum((r_[mask] - r_[mask].mean()) ** 2))
            r2 = 1 - ss_res / max(ss_tot, 1e-12)
            return float(slope), float(r2)

        betas_r[delta], r2_r[delta] = _lin(I_r, r_r)
        betas_s[delta], r2_s[delta] = _lin(I_s, r_s)
        pass_sign[delta] = (betas_r[delta] > 0 and betas_s[delta] > 0)

    return CondReturnResult(
        deltas=list(deltas),
        bin_centers=centers,
        real_means=real_means,
        synth_means=synth_means,
        real_bands=bands,
        betas_real=betas_r,
        betas_synth=betas_s,
        r2_real=r2_r,
        r2_synth=r2_s,
        pass_sign_per_delta=pass_sign,
    )


@dataclass
class JointResult:
    real_pmf: np.ndarray                  # (n_spread_bins, n_imb_bins)
    synth_pmf: np.ndarray
    spread_labels: list[str]
    imb_labels: list[str]
    jensen_shannon: float
    pass_jsd: bool


def joint_spread_imbalance(
    real_concat: pl.DataFrame,
    synth_concat: pl.DataFrame,
    tick_size_usd: float = 0.01,
) -> JointResult:
    """D.2 — joint distribution of (spread, imbalance)."""
    spread_bins = np.array([0, 1.5, 2.5, 3.5, 4.5, np.inf]) * tick_size_usd
    imb_bins = np.linspace(-1.0, 1.0, 6)
    spread_labels = ["1c", "2c", "3c", "4c", ">=5c"]
    imb_labels = ["[-1,-.6]", "[-.6,-.2]", "[-.2,.2]", "[.2,.6]", "[.6,1]"]

    def _pmf(df):
        sp = df["spread"].to_numpy()
        im = _imbalance(df)
        h, _, _ = np.histogram2d(sp, im, bins=[spread_bins, imb_bins])
        return h / max(h.sum(), 1e-9)

    real_pmf = _pmf(real_concat)
    synth_pmf = _pmf(synth_concat)
    jsd = float(jensenshannon(real_pmf.ravel() + 1e-12, synth_pmf.ravel() + 1e-12))
    return JointResult(
        real_pmf=real_pmf,
        synth_pmf=synth_pmf,
        spread_labels=spread_labels,
        imb_labels=imb_labels,
        jensen_shannon=jsd,
        pass_jsd=jsd < 0.3,
    )


@dataclass
class CondMomentsResult:
    table: pd.DataFrame                           # rows=regime, cols=stat
    z_scores: pd.DataFrame
    monotonicity_flags: dict[str, bool]
    pass_majority_z: bool


def conditional_moments_by_regime(
    real_per_day: list[pl.DataFrame],
    synth_concat: pl.DataFrame,
) -> CondMomentsResult:
    """D.3 — per-regime conditional moments (tests whether conditioning works)."""
    regimes = ("base", "high_vol", "toxic", "thin")
    stats_names = ("mean_abs_return", "mean_spread", "std_return", "trade_fraction", "mean_bid_sz")

    # Real: aggregate per-day frames, partition by regime_label per row
    real_concat = pl.concat(real_per_day) if real_per_day else pl.DataFrame()

    def _stats_on(df: pl.DataFrame) -> dict:
        if df.height == 0:
            return {k: float("nan") for k in stats_names}
        r = df["mid_return"].fill_null(0).to_numpy()
        return {
            "mean_abs_return": float(np.mean(np.abs(r))),
            "mean_spread": float(df["spread"].mean()),
            "std_return": float(np.std(r, ddof=1)) if r.size > 1 else 0.0,
            "trade_fraction": float((df["event_type"] == "trade").mean()),
            "mean_bid_sz": float(df["bid_sz"].mean()),
        }

    real_means: dict[str, dict] = {}
    real_stds: dict[str, dict] = {}
    synth_means: dict[str, dict] = {}

    for r in regimes:
        # Real: filter per-day, then aggregate
        real_day_stats: list[dict] = []
        for d in real_per_day:
            sub = d.filter(pl.col("regime_label") == r) if "regime_label" in d.columns else d
            if sub.height:
                real_day_stats.append(_stats_on(sub))
        if real_day_stats:
            df_d = pd.DataFrame(real_day_stats)
            real_means[r] = df_d.mean().to_dict()
            real_stds[r] = df_d.std(ddof=1).fillna(0).to_dict()
        else:
            real_means[r] = {k: float("nan") for k in stats_names}
            real_stds[r] = {k: float("nan") for k in stats_names}
        # Synth: filter
        sub_s = synth_concat.filter(pl.col("regime_label") == r) if "regime_label" in synth_concat.columns else synth_concat
        synth_means[r] = _stats_on(sub_s)

    table_rows = []
    for r in regimes:
        for s in stats_names:
            table_rows.append({
                "regime": r, "stat": s,
                "real_mean": real_means[r][s],
                "real_std": real_stds[r][s],
                "synth_mean": synth_means[r][s],
            })
    table = pd.DataFrame(table_rows)

    z_rows = []
    for r in regimes:
        z_row = {"regime": r}
        for s in stats_names:
            denom = max(abs(real_stds[r][s]), 1e-9)
            z_row[s] = (synth_means[r][s] - real_means[r][s]) / denom if not np.isnan(real_stds[r][s]) else float("nan")
        z_rows.append(z_row)
    z_df = pd.DataFrame(z_rows).set_index("regime")

    # Monotonicity test: for each statistic, does it shift in the expected direction across regimes?
    # base → high_vol: mean_abs_return should increase
    # base → toxic: trade_fraction should increase
    mono = {}
    if all(r in synth_means for r in regimes):
        mono["mean_abs_return_high_vol_gt_base"] = (
            synth_means["high_vol"]["mean_abs_return"] > synth_means["base"]["mean_abs_return"]
        )
        mono["trade_fraction_toxic_gt_base"] = (
            synth_means["toxic"]["trade_fraction"] > synth_means["base"]["trade_fraction"]
        )

    z_finite = z_df.values[np.isfinite(z_df.values)]
    pass_majority = bool(np.mean(np.abs(z_finite) < 2.0) > 0.5) if z_finite.size else False

    return CondMomentsResult(
        table=table,
        z_scores=z_df,
        monotonicity_flags=mono,
        pass_majority_z=pass_majority,
    )


# ─── Pass-fail summary ─────────────────────────────────────────────────

def consolidated_summary(
    return_stats: ReturnStats,
    spread_stats: SpreadStats,
    size_stats: SizeStats,
    event_stats: EventCompositionStats,
    acf_abs: ACFResult,
    acf_sign: ACFResult,
    acf_ret: ACFResult,
    rv: RVSignatureResult,
    cond_ret: CondReturnResult,
    joint: JointResult,
    cond_mom: CondMomentsResult,
    crossed_books_rate: float = 0.0,
) -> pd.DataFrame:
    """G.1 — Pass/Fail summary table."""
    rows = [
        ("1", "Return W-1", "B.1", return_stats.wasserstein_1, "< 2× inter-day", return_stats.pass_w1),
        ("2", "Return excess kurtosis", "B.1", return_stats.synth["kurt"], "> 0", return_stats.pass_kurtosis),
        ("3", "Spread 1-tick fraction", "B.2", spread_stats.synth_one_tick_frac, "> 0.80", spread_stats.pass_one_tick),
        ("4", "Spread W-1", "B.2", spread_stats.wasserstein_1, "< 2× inter-day", spread_stats.pass_w1),
        ("5", "Size tail exponent", "B.3",
            f"real={size_stats.tail_alpha_real:.3f}, synth={size_stats.tail_alpha_synth:.3f}",
            "within ±30% of real", size_stats.pass_tail),
        ("6", "Trade fraction (synth vs real)", "B.4",
            f"real={event_stats.real_trade_frac:.3f}, synth={event_stats.synth_trade_frac:.3f}",
            "~10% real (report actual)", None),
        ("7", "ACF(|r|) lag-50 deviation", "C.1", acf_abs.lag_50_deviation, "< 0.5", acf_abs.pass_lag50),
        ("8", "ACF(|r|) power-law β",       "C.1",
            f"real={acf_abs.power_law_beta_real:.3f}, synth={acf_abs.power_law_beta_synth:.3f}",
            "within ±50%", acf_abs.pass_power_law),
        ("9", "Trade-sign lag-1 ACF sign",   "C.2",
            f"real={acf_sign.real_concat[1]:.3f}, synth={acf_sign.synth_concat[1]:.3f}",
            "both positive", acf_sign.pass_lag50),
        ("10", "Return lag-1 ACF",            "C.3", acf_ret.synth_concat[1], "≤ 0", True),
        ("11", "RV signature shape",          "C.4",
            f"ratio_real={rv.ratio_real:.3f}, ratio_synth={rv.ratio_synth:.3f}",
            "ratio within 2×", rv.pass_shape),
        ("12", "E[r|I] sign at Δ=10",        "D.1",
            f"β_real={cond_ret.betas_real.get(10, np.nan):.3e}, β_synth={cond_ret.betas_synth.get(10, np.nan):.3e}",
            "β_synth > 0", cond_ret.pass_sign_per_delta.get(10, False)),
        ("13", "E[r|I] sign at Δ=50",        "D.1",
            f"β_real={cond_ret.betas_real.get(50, np.nan):.3e}, β_synth={cond_ret.betas_synth.get(50, np.nan):.3e}",
            "β_synth > 0", cond_ret.pass_sign_per_delta.get(50, False)),
        ("14", "Joint spread-imbalance JSD",  "D.2", joint.jensen_shannon, "< 0.3", joint.pass_jsd),
        ("15", "Conditional moments (|z|<2)", "D.3", "majority", "> 50% of cells", cond_mom.pass_majority_z),
        ("16", "Crossed books rate (decode)", "—", crossed_books_rate, "< 1e-5", crossed_books_rate < 1e-5),
    ]
    return pd.DataFrame(rows, columns=["#", "Check", "Section", "Value", "Criterion", "Pass"])
