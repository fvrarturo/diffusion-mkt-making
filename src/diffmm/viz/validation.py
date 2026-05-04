"""Validation figures for the Work5 generator-validation suite.

Each function takes a result dataclass from `diffmm.eval.stylized` or
`diffmm.eval.model_diagnostics` and produces one PDF + PNG. Figures match
the Work5 plan's panel layouts.

All functions accept `out_path` (without extension); both .pdf and .png are
written.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ..eval import stylized, model_diagnostics

REAL_COLOR = "#1f77b4"
SYNTH_COLOR = "#ff7f0e"
BAND_ALPHA = 0.20


def _save(fig, out_path: str | Path) -> None:
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(p.with_suffix(".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ─── Part B ────────────────────────────────────────────────────────────

def plot_return_distribution(stats: stylized.ReturnStats, out_path: str | Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    r, s = stats.real_values, stats.synth_values
    if r.size == 0 or s.size == 0:
        for ax in axes.ravel():
            ax.text(0.5, 0.5, "no data", ha="center")
        _save(fig, out_path); return

    # (a) Histogram overlay (log y)
    ax = axes[0, 0]
    lo, hi = -0.005, 0.005
    bins = np.linspace(lo, hi, 200)
    ax.hist(np.clip(r, lo, hi), bins=bins, density=True, alpha=0.55,
            color=REAL_COLOR, label=f"real (kurt={stats.real['kurt']:.2f})")
    ax.hist(np.clip(s, lo, hi), bins=bins, density=True, alpha=0.55,
            color=SYNTH_COLOR, label=f"synth (kurt={stats.synth['kurt']:.2f})")
    ax.set_yscale("log")
    ax.set_title("(a) mid_return distribution (log y)")
    ax.set_xlabel("mid_return"); ax.set_ylabel("density")
    ax.legend()

    # (b) QQ plot
    ax = axes[0, 1]
    probs = np.linspace(0.001, 0.999, 1000)
    real_q = np.quantile(r, probs)
    synth_q = np.quantile(s, probs)
    ax.plot(real_q, synth_q, lw=1.5, color=SYNTH_COLOR)
    lo_, hi_ = min(real_q.min(), synth_q.min()), max(real_q.max(), synth_q.max())
    ax.plot([lo_, hi_], [lo_, hi_], "k--", lw=0.8, label="identity")
    ax.set_title("(b) QQ plot — real vs synth quantiles")
    ax.set_xlabel("real quantile"); ax.set_ylabel("synth quantile")
    ax.legend()

    # (c) Excess kurtosis bars
    ax = axes[1, 0]
    bars = ax.bar(["real", "synth"], [stats.real["kurt"], stats.synth["kurt"]],
                  color=[REAL_COLOR, SYNTH_COLOR])
    for b, v in zip(bars, [stats.real["kurt"], stats.synth["kurt"]]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center",
                va="bottom" if v >= 0 else "top")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_title("(c) Excess kurtosis (>0 = leptokurtic)")
    ax.set_ylabel("kurtosis (Fisher)")

    # (d) W-1 with inter-day reference band
    ax = axes[1, 1]
    band_lo = max(0, stats.inter_day_w1_mean - 2 * stats.inter_day_w1_std)
    band_hi = stats.inter_day_w1_mean + 2 * stats.inter_day_w1_std
    ax.axhspan(band_lo, band_hi, alpha=0.20, color=REAL_COLOR,
               label=f"inter-day band [{band_lo:.2e}, {band_hi:.2e}]")
    ax.bar(["real-vs-synth"], [stats.wasserstein_1], color=SYNTH_COLOR)
    ax.text(0, stats.wasserstein_1, f"{stats.wasserstein_1:.2e}", ha="center", va="bottom")
    ax.set_title(f"(d) Wasserstein-1 — pass: {stats.pass_w1}")
    ax.set_ylabel("W-1")
    ax.legend()

    fig.suptitle(f"B.1 Return distribution    KS={stats.ks_statistic:.4f} (p={stats.ks_pvalue:.2e})")
    fig.tight_layout()
    _save(fig, out_path)


def plot_spread_distribution(stats: stylized.SpreadStats, out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) PMF bars side-by-side
    ax = axes[0]
    labels = list(stats.real_pmf.keys())
    x = np.arange(len(labels))
    width = 0.4
    ax.bar(x - width / 2, [stats.real_pmf[k] for k in labels], width,
           color=REAL_COLOR, label="real")
    ax.bar(x + width / 2, [stats.synth_pmf[k] for k in labels], width,
           color=SYNTH_COLOR, label="synth")
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_title(f"(a) Spread PMF  (1-tick: real {stats.real_one_tick_frac:.1%}, "
                 f"synth {stats.synth_one_tick_frac:.1%})")
    ax.set_ylabel("fraction of events"); ax.legend()

    # (b) CDF overlay
    ax = axes[1]
    if stats.real_values.size and stats.synth_values.size:
        srt_r = np.sort(stats.real_values)
        srt_s = np.sort(stats.synth_values)
        ax.step(srt_r, np.arange(srt_r.size) / srt_r.size, color=REAL_COLOR, label="real")
        ax.step(srt_s, np.arange(srt_s.size) / srt_s.size, color=SYNTH_COLOR, label="synth")
    ax.set_xlim(0, 0.06)
    ax.set_title("(b) Spread CDF")
    ax.set_xlabel("spread (USD)"); ax.set_ylabel("CDF"); ax.legend()

    # (c) W-1
    ax = axes[2]
    band_lo = 0
    band_hi = 2 * stats.inter_day_w1_mean
    ax.axhspan(band_lo, band_hi, alpha=0.20, color=REAL_COLOR,
               label=f"2× inter-day = {band_hi:.2e}")
    ax.bar(["real-vs-synth"], [stats.wasserstein_1], color=SYNTH_COLOR)
    ax.text(0, stats.wasserstein_1, f"{stats.wasserstein_1:.2e}", ha="center", va="bottom")
    ax.set_title(f"(c) Spread W-1 — pass: {stats.pass_w1}")
    ax.set_ylabel("W-1"); ax.legend()

    fig.suptitle("B.2 Spread distribution")
    fig.tight_layout()
    _save(fig, out_path)


def plot_size_distribution(stats: stylized.SizeStats, out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # (a) Log-log CCDF
    ax = axes[0]
    for arr, color, lbl in ((stats.real_log_values, REAL_COLOR, "real"),
                             (stats.synth_log_values, SYNTH_COLOR, "synth")):
        # CCDF in raw size space
        x_raw = np.exp(arr) - 1
        x_raw = x_raw[x_raw > 0]
        if x_raw.size:
            srt = np.sort(x_raw)
            ccdf = 1 - np.arange(srt.size) / srt.size
            ax.loglog(srt, ccdf, color=color, label=lbl, lw=1.2)
    ax.set_title(f"(a) bid_sz CCDF  (α: real={stats.tail_alpha_real:.2f}, "
                 f"synth={stats.tail_alpha_synth:.2f})")
    ax.set_xlabel("size"); ax.set_ylabel("P(X > size)"); ax.legend()

    # (b) QQ on log sizes
    ax = axes[1]
    if stats.real_log_values.size and stats.synth_log_values.size:
        probs = np.linspace(0.01, 0.99, 200)
        rq = np.quantile(stats.real_log_values, probs)
        sq = np.quantile(stats.synth_log_values, probs)
        ax.plot(rq, sq, color=SYNTH_COLOR)
        lo, hi = min(rq.min(), sq.min()), max(rq.max(), sq.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
    ax.set_title(f"(b) QQ on log(bid_sz)  pass W-1: {stats.pass_w1}")
    ax.set_xlabel("real log(size+1)"); ax.set_ylabel("synth log(size+1)")

    fig.suptitle("B.3 Size distribution")
    fig.tight_layout()
    _save(fig, out_path)


def plot_event_composition(stats: stylized.EventCompositionStats, out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # (a) Trade-fraction overall
    ax = axes[0]
    bars = ax.bar(["real", "synth"], [stats.real_trade_frac, stats.synth_trade_frac],
                  color=[REAL_COLOR, SYNTH_COLOR])
    for b, v in zip(bars, [stats.real_trade_frac, stats.synth_trade_frac]):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.1%}", ha="center", va="bottom")
    ax.set_title("(a) Trade fraction overall")
    ax.set_ylabel("trades / events"); ax.set_ylim(0, max(0.7, stats.synth_trade_frac * 1.2))

    # (b) Trade-fraction by regime
    ax = axes[1]
    regimes = list(stats.synth_trade_frac_by_regime.keys())
    if regimes:
        vals = [stats.synth_trade_frac_by_regime[r] for r in regimes]
        ax.bar(regimes, vals, color=SYNTH_COLOR)
        for r, v in zip(regimes, vals):
            ax.text(r, v, f"{v:.1%}", ha="center", va="bottom")
    ax.set_title("(b) Trade fraction by synth regime")
    ax.set_ylabel("synth trades / events")

    # (c) Inter-trade-gap histogram
    ax = axes[2]
    if stats.real_inter_trade_gap.size and stats.synth_inter_trade_gap.size:
        bins = np.linspace(0, 100, 50)
        ax.hist(np.clip(stats.real_inter_trade_gap, 0, 100), bins=bins, density=True,
                alpha=0.55, color=REAL_COLOR, label="real")
        ax.hist(np.clip(stats.synth_inter_trade_gap, 0, 100), bins=bins, density=True,
                alpha=0.55, color=SYNTH_COLOR, label="synth")
    ax.set_yscale("log")
    ax.set_title("(c) Inter-trade event gap")
    ax.set_xlabel("events between trades"); ax.set_ylabel("density"); ax.legend()

    fig.suptitle("B.4 Event composition")
    fig.tight_layout()
    _save(fig, out_path)


# ─── Part C ────────────────────────────────────────────────────────────

def plot_acf(result: stylized.ACFResult, out_path: str | Path,
             include_loglog_inset: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    lags = np.arange(result.max_lag + 1)
    ax.plot(lags, result.real_concat, color=REAL_COLOR, lw=1.5, label="real")
    ax.fill_between(lags, result.real_band_lower, result.real_band_upper,
                    color=REAL_COLOR, alpha=BAND_ALPHA, label="real ±2σ band")
    ax.plot(lags, result.synth_concat, color=SYNTH_COLOR, lw=1.5, label="synth")
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlabel("lag"); ax.set_ylabel(f"ACF[{result.name}]")
    title = f"{result.name} ACF"
    if not np.isnan(result.lag_50_deviation):
        title += f"   lag-50 dev: {result.lag_50_deviation:.2f}"
    title += f"  pass: {result.pass_lag50}"
    ax.set_title(title)
    ax.legend()
    if include_loglog_inset and not np.isnan(result.power_law_beta_real):
        ax_in = ax.inset_axes([0.55, 0.55, 0.4, 0.4])
        valid_real = (lags >= 10) & np.isfinite(result.real_concat) & (result.real_concat > 0)
        valid_synth = (lags >= 10) & np.isfinite(result.synth_concat) & (result.synth_concat > 0)
        if valid_real.sum() > 5:
            ax_in.loglog(lags[valid_real], result.real_concat[valid_real],
                         "o-", color=REAL_COLOR, markersize=2,
                         label=f"β={result.power_law_beta_real:.2f}")
        if valid_synth.sum() > 5:
            ax_in.loglog(lags[valid_synth], result.synth_concat[valid_synth],
                         "o-", color=SYNTH_COLOR, markersize=2,
                         label=f"β={result.power_law_beta_synth:.2f}")
        ax_in.set_title("log-log decay", fontsize=8)
        ax_in.legend(fontsize=7)
    fig.tight_layout()
    _save(fig, out_path)


def plot_rv_signature(rv: stylized.RVSignatureResult, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.errorbar(rv.deltas, rv.real_mean,
                yerr=[rv.real_mean - rv.real_band_low, rv.real_band_high - rv.real_mean],
                color=REAL_COLOR, marker="o", label="real")
    ax.errorbar(rv.deltas, rv.synth_mean,
                yerr=[rv.synth_mean - rv.synth_band_low, rv.synth_band_high - rv.synth_mean],
                color=SYNTH_COLOR, marker="s", label="synth")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("sampling stride (events)")
    ax.set_ylabel("realized variance per event (log)")
    ax.set_title(f"C.4 RV signature   ratio_real={rv.ratio_real:.2f}, "
                 f"ratio_synth={rv.ratio_synth:.2f}, pass: {rv.pass_shape}")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


# ─── Part D ────────────────────────────────────────────────────────────

def plot_conditional_return_imbalance(res: stylized.CondReturnResult,
                                       out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, len(res.deltas), figsize=(5 * len(res.deltas), 4.5),
                             squeeze=False)
    for i, delta in enumerate(res.deltas):
        ax = axes[0, i]
        ax.plot(res.bin_centers, res.real_means[delta], color=REAL_COLOR, marker="o",
                label=f"real β={res.betas_real[delta]:.2e}, R²={res.r2_real[delta]:.3f}")
        lo, hi = res.real_bands[delta]
        ax.fill_between(res.bin_centers, lo, hi, color=REAL_COLOR, alpha=BAND_ALPHA)
        ax.plot(res.bin_centers, res.synth_means[delta], color=SYNTH_COLOR, marker="s",
                label=f"synth β={res.betas_synth[delta]:.2e}, R²={res.r2_synth[delta]:.3f}")
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("imbalance bin center"); ax.set_ylabel(f"E[r_{{t+{delta}}} | I_t]")
        ax.set_title(f"Δ={delta}  pass: {res.pass_sign_per_delta[delta]}")
        ax.legend(fontsize=8)
    fig.suptitle("D.1 Conditional return on imbalance (the OFI→return signal)")
    fig.tight_layout()
    _save(fig, out_path)


def plot_joint_spread_imbalance(j: stylized.JointResult, out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    vmax = max(j.real_pmf.max(), j.synth_pmf.max())
    for ax, pmf, title in zip(axes, [j.real_pmf, j.synth_pmf], ["real", "synth"]):
        im = ax.imshow(pmf, aspect="auto", origin="lower", cmap="viridis",
                       vmin=0, vmax=vmax)
        ax.set_xticks(range(len(j.imb_labels)));  ax.set_xticklabels(j.imb_labels, rotation=30)
        ax.set_yticks(range(len(j.spread_labels))); ax.set_yticklabels(j.spread_labels)
        ax.set_xlabel("imbalance"); ax.set_ylabel("spread")
        ax.set_title(title)
        for i in range(pmf.shape[0]):
            for k in range(pmf.shape[1]):
                ax.text(k, i, f"{pmf[i,k]:.2f}", ha="center", va="center",
                        color="white" if pmf[i,k] < vmax / 2 else "black", fontsize=7)
    fig.suptitle(f"D.2 Joint spread×imbalance — JSD={j.jensen_shannon:.3f}, pass: {j.pass_jsd}")
    fig.colorbar(im, ax=axes, fraction=0.04)
    _save(fig, out_path)


def plot_conditional_moments(res: stylized.CondMomentsResult, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    z = res.z_scores.copy()
    cmap = plt.get_cmap("RdYlGn_r")
    im = ax.imshow(np.abs(z.values), aspect="auto", cmap=cmap, vmin=0, vmax=4)
    ax.set_xticks(range(z.shape[1])); ax.set_xticklabels(z.columns, rotation=20, ha="right")
    ax.set_yticks(range(z.shape[0])); ax.set_yticklabels(z.index)
    for i in range(z.shape[0]):
        for k in range(z.shape[1]):
            v = z.values[i, k]
            ax.text(k, i, f"{v:+.2f}", ha="center", va="center",
                    color="white" if abs(v) > 2 else "black", fontsize=9)
    ax.set_title(f"D.3 Conditional moments z-scores (|z|<2 = pass)  "
                 f"majority pass: {res.pass_majority_z}")
    fig.colorbar(im, ax=ax, label="|z|")
    fig.tight_layout()
    _save(fig, out_path)


# ─── Part E ────────────────────────────────────────────────────────────

def plot_training_curves(v1_steps: pd.DataFrame, v2_steps: pd.DataFrame,
                         out_path: str | Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    if not v1_steps.empty:
        ax.plot(v1_steps["step"], v1_steps["loss"], color=REAL_COLOR,
                lw=0.8, alpha=0.7, label="v1")
    if not v2_steps.empty:
        ax.plot(v2_steps["step"], v2_steps["loss"], color=SYNTH_COLOR,
                lw=0.8, alpha=0.9, label="v2 (Min-SNR)")
    ax.set_yscale("log")
    ax.set_xlabel("training step"); ax.set_ylabel("train loss (log)")
    ax.set_title("(a) Training loss vs step"); ax.legend()

    ax = axes[1]
    # Best vals annotated
    for label, color, val_loss in (("v1", REAL_COLOR, 0.0160), ("v2", SYNTH_COLOR, 0.0106)):
        ax.bar(label, val_loss, color=color)
        ax.text(label, val_loss, f"{val_loss:.4f}", ha="center", va="bottom")
    ax.set_title("(b) Best val loss")
    ax.set_ylabel("val/loss")
    fig.suptitle("E.1 Training curves")
    fig.tight_layout()
    _save(fig, out_path)


def plot_per_timestep_quality(v1: model_diagnostics.PerTimestepResult | None,
                                v2: model_diagnostics.PerTimestepResult,
                                out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    ax2 = ax.twinx()
    if v1 is not None:
        ax.errorbar(v1.timesteps, v1.mse_mean, yerr=v1.mse_std,
                    color=REAL_COLOR, marker="o", linestyle="--",
                    label="v1 MSE", capsize=3, alpha=0.7)
    ax.errorbar(v2.timesteps, v2.mse_mean, yerr=v2.mse_std,
                color=SYNTH_COLOR, marker="s", label="v2 MSE", capsize=3)
    ax.axvspan(200, 600, alpha=0.10, color="grey", label="critical range t∈[200,600]")
    ax.set_xlabel("diffusion timestep t"); ax.set_ylabel("ε-prediction MSE")
    ax2.plot(v2.timesteps, v2.min_snr_weight, color="black", lw=0.8, linestyle=":",
             label="Min-SNR weight (v2)")
    ax2.set_ylabel("Min-SNR weight")
    ax2.set_ylim(0, 1.1)
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    ax.set_title("E.2 Per-timestep denoising quality")
    fig.tight_layout()
    _save(fig, out_path)


def plot_x0_clip_activation(v2: model_diagnostics.ClipActivationResult,
                              v1: model_diagnostics.ClipActivationResult | None,
                              out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    if v1 is not None:
        ax.plot(v1.per_step_rate, color=REAL_COLOR, lw=0.8, linestyle="--",
                label=f"v1 (overall {v1.overall_rate:.2%})")
    ax.plot(v2.per_step_rate, color=SYNTH_COLOR, lw=1.0,
            label=f"v2 (overall {v2.overall_rate:.2%})")
    ax.set_xlabel("DDIM step (T-1 → 0)"); ax.set_ylabel("|x0_pred| > clip rate")
    ax.set_title("E.3 x0_clip activation rate")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def plot_noise_schedule(ns: model_diagnostics.NoiseScheduleResult, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(ns.t, ns.sqrt_alpha_bar, color=REAL_COLOR, label="√α̅ (signal scale)")
    ax.plot(ns.t, ns.sqrt_one_minus_alpha_bar, color=SYNTH_COLOR, label="√(1-α̅) (noise scale)")
    ax.set_xlabel("t"); ax.set_ylabel("scale")
    ax2 = ax.twinx()
    ax2.semilogy(ns.t, ns.snr, color="black", lw=0.8, linestyle=":", label="SNR(t) (log)")
    ax2.set_ylabel("SNR (log)")
    ax.set_title("E.4 Noise schedule")
    ax.legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.tight_layout()
    _save(fig, out_path)


# ─── Part F ────────────────────────────────────────────────────────────

def plot_regime_embeddings(res: model_diagnostics.EmbeddingPCAResult,
                            out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 6))
    color_map = {"base": "grey", "high_vol": "red", "toxic": "purple",
                 "thin": "blue", "null": "black"}
    for label in set(res.regime_labels):
        idx = [i for i, l in enumerate(res.regime_labels) if l == label]
        marker = "*" if label == "null" else "o"
        size = 200 if label == "null" else 35
        ax.scatter(res.points_2d[idx, 0], res.points_2d[idx, 1],
                   c=color_map.get(label, "grey"), label=label, marker=marker,
                   s=size, alpha=0.7, edgecolors="black", linewidths=0.5)
    ax.set_xlabel(f"PC1 ({res.var_explained[0]:.1%})")
    ax.set_ylabel(f"PC2 ({res.var_explained[1]:.1%})")
    ax.set_title(f"F.1 Regime embedding PCA — total var {sum(res.var_explained):.1%}")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)


def plot_regime_embedding_cosine(res: model_diagnostics.EmbeddingCosineResult,
                                   out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    M = res.matrix
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    ax.set_xticks(range(len(res.labels))); ax.set_xticklabels(res.labels, rotation=45)
    ax.set_yticks(range(len(res.labels))); ax.set_yticklabels(res.labels)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                    color="white" if abs(M[i, j]) > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title("F.1b Regime embedding cosine similarity")
    fig.text(0.5, -0.02, res.diagnosis, ha="center", va="top",
             fontsize=8, wrap=True, style="italic")
    fig.tight_layout()
    _save(fig, out_path)


def plot_film_modulation(film: model_diagnostics.FiLMResult, out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    df = film.table.set_index("regime")
    df.plot.bar(ax=ax)
    ax.set_title("F.2 FiLM γ deviation and β norm by regime")
    ax.set_ylabel("magnitude")
    fig.tight_layout()
    _save(fig, out_path)


def plot_attention_maps(att: model_diagnostics.AttentionMapsResult,
                         out_path: str | Path) -> None:
    keys = att.titles
    n = len(keys)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 4 * rows), squeeze=False)
    for i, key in enumerate(keys):
        ax = axes[i // cols, i % cols]
        m = att.maps[key]
        im = ax.imshow(m, cmap="viridis", aspect="auto")
        ax.set_title(key)
        ax.set_xlabel("attended event"); ax.set_ylabel("query event")
        fig.colorbar(im, ax=ax, fraction=0.04)
    for j in range(n, rows * cols):
        axes[j // cols, j % cols].axis("off")
    fig.suptitle("F.3 Attention weights — block 0, head 0")
    fig.tight_layout()
    _save(fig, out_path)


def plot_guidance_sweep(gs: model_diagnostics.GuidanceSweepResult, out_path: str | Path) -> None:
    stats_keys = ["mean_abs_return", "mean_spread", "std_return", "trade_fraction", "mean_bid_sz"]
    fig, axes = plt.subplots(1, len(stats_keys), figsize=(4 * len(stats_keys), 4),
                             squeeze=False)
    if gs.table.empty:
        return
    ws = gs.table["w"].values
    for i, k in enumerate(stats_keys):
        ax = axes[0, i]
        means = gs.table[f"{k}_mean"].values
        stds = gs.table[f"{k}_std"].values
        ax.errorbar(ws, means, yerr=stds, color=SYNTH_COLOR, marker="o",
                    label="synth")
        target = gs.real_targets.get(k)
        if target is not None:
            ax.axhline(target, color=REAL_COLOR, linestyle="--",
                       label=f"real target = {target:.3e}")
        ax.axvline(1.0, color="black", linestyle=":", alpha=0.4, label="default w=1")
        ax.set_xlabel("guidance weight w"); ax.set_ylabel(k)
        ax.set_title(k); ax.legend(fontsize=7)
    fig.suptitle("F.4 Guidance weight sensitivity")
    fig.tight_layout()
    _save(fig, out_path)


def plot_per_feature_loss(res: model_diagnostics.PerFeatureMSEResult,
                           out_path: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.bar(res.feature_names, res.mse_per_feature, color=SYNTH_COLOR)
    for b, v in zip(bars, res.mse_per_feature):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(1.0, color="black", linestyle=":", alpha=0.5, label="ideal MSE = 1")
    ax.set_xticklabels(res.feature_names, rotation=30, ha="right")
    ax.set_ylabel("MSE per feature at t=300")
    ax.set_title("F.5 Per-feature ε-prediction MSE")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_path)
