#!/usr/bin/env python3
"""
Ranking-mechanism diagnostic — what actually drives the A1/A2 ranking
under the central hypothesis test?

Motivation
----------
The OFI→return regression has R² = 0.0001 on REAL INTC val data (deep
diagnostic, 2026-05-07), so the FINAL §8 narrative — "v-prediction breaks
OFI→return correlation, causing the A1/A2 swap" — may be explaining noise.
We can't predict whether v9 will preserve the ranking unless we know
which agent metric *actually* differs between A1 (AS, no OFI) and A2
(AS+OFI) in real backtests.

Two analyses on existing pipeline outputs (CPU only, ~5-10 min):

Part 1: Per-agent metric decomposition.
    Uses the per-day metrics.csv that ph4_replay_real / replay_synth /
    open_holdout already wrote. Each row = (tape, agent, terminal_pnl,
    n_fills, fill_rate, inventory_var, pnl_max_drawdown). We compute
    per-agent aggregates and A2−A1 deltas, then ask: across real / truth /
    v2 / v7_b / v8 / v8_b, which dimension is the dominant separator?
    If it's `n_fills` or `fill_rate` (not `terminal_pnl` directly), the
    A1/A2 gap is a fill-rate / inventory mechanism, not a return-prediction
    mechanism — which reframes what v9 needs to succeed at.

Part 2: Conditional-on-trade return distribution.
    Loads decoded parquets and compares frac_zero + the conditional
    (nonzero) return distribution across models. v2 has frac_zero ≈ 0.0
    (every event has a tiny continuous return); real has frac_zero = 0.89.
    If v2 nonetheless gets ρ_diff = +0.40, zero-inflation doesn't matter
    for the ranking test — and the dequantization fix in v9 buys validation-
    suite realism, not ρ_diff per se.

Run (cluster, CPU):
    python scripts/61_ranking_mechanism.py \
        --metrics \
            truth:results/backtests/holdout/INTC/metrics.csv \
            real:results/backtests/real/INTC/metrics.csv \
            v2:results/_archive/v8/synthetic_backtests/metrics_v2.csv \
            v7_b:results/_archive/v8/synthetic_backtests/metrics_v7_b.csv \
            v8:results/_archive/v8/synthetic_backtests/metrics.csv \
            v8_b:results/_archive/v8_b/synthetic_backtests/metrics.csv \
        --real-dir data/processed/val \
        --synth-dirs \
            v2:$HOME/orcd/scratch/diffmm/synthetic_v2_stitched/INTC \
            v7_b:$HOME/orcd/scratch/diffmm/synthetic_v7_b_stitched/INTC \
            v8:$HOME/orcd/scratch/diffmm/synthetic/intc_v8/INTC \
            v8_b:$HOME/orcd/scratch/diffmm/synthetic/intc_v8_b/INTC \
        --out results/ranking_mechanism

Outputs:
    {out}/agent_decomposition.csv         per-(model, agent) aggregates
    {out}/a2_minus_a1_deltas.csv          ΔA2−A1 per metric per model
    {out}/conditional_returns.csv         frac_zero + conditional stats
    {out}/summary.txt                     human-readable headline report
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# Match column conventions from src/diffmm/sim/runner.py DayMetrics
METRIC_COLS = [
    "terminal_pnl",
    "n_fills",
    "fill_rate",
    "inventory_var",
    "pnl_max_drawdown",
]

# Agent name canonicalization — handle slight casing/format drift across runs
AGENT_CANON = {
    "A0": "A0_const", "A0_const": "A0_const",
    "A1": "A1_AS", "A1_AS": "A1_AS",
    "A2": "A2_AS_OFI", "A2_AS_OFI": "A2_AS_OFI",
    "A3": "A3_AS_VPIN", "A3_AS_VPIN": "A3_AS_VPIN",
}


def _canon_agent(name: str) -> str:
    return AGENT_CANON.get(str(name).strip(), str(name).strip())


def _parse_kv(items: list[str]) -> dict[str, str]:
    """Parse `label:path` pairs into a dict."""
    out: dict[str, str] = {}
    for it in items:
        if ":" not in it:
            raise ValueError(f"expected label:path, got {it!r}")
        label, path = it.split(":", 1)
        out[label.strip()] = path.strip()
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PART 1: per-agent metric decomposition
# ─────────────────────────────────────────────────────────────────────────────


def load_metrics_csv(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "agent" not in df.columns:
        raise ValueError(f"{path}: no 'agent' column; have {list(df.columns)}")
    df["agent"] = df["agent"].map(_canon_agent)
    # Tolerate alternative column names from older pipeline versions
    rename = {
        "terminal_PNL": "terminal_pnl",
        "fills": "n_fills",
        "fillrate": "fill_rate",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    return df


def aggregate_per_agent(df: pd.DataFrame) -> pd.DataFrame:
    """Per-(agent) aggregates of each metric. Returns one row per agent."""
    rows = []
    for agent, sub in df.groupby("agent"):
        row: dict[str, object] = {"agent": agent, "n_days": len(sub)}
        for col in METRIC_COLS:
            if col in sub.columns:
                vals = pd.to_numeric(sub[col], errors="coerce").dropna().values
                if len(vals) > 0:
                    row[f"{col}_mean"] = float(np.mean(vals))
                    row[f"{col}_std"] = float(np.std(vals))
                    row[f"{col}_median"] = float(np.median(vals))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("agent").reset_index(drop=True)


def a2_minus_a1_deltas(per_agent: pd.DataFrame) -> dict[str, float]:
    """For each metric, compute A2_AS_OFI mean minus A1_AS mean."""
    a1 = per_agent[per_agent["agent"] == "A1_AS"]
    a2 = per_agent[per_agent["agent"] == "A2_AS_OFI"]
    if len(a1) != 1 or len(a2) != 1:
        return {}
    a1, a2 = a1.iloc[0], a2.iloc[0]
    out: dict[str, float] = {}
    for col in METRIC_COLS:
        key = f"{col}_mean"
        if key in a1 and key in a2:
            try:
                out[col] = float(a2[key]) - float(a1[key])
                # Also normalized: Δ / σ_A1 (effect size)
                std_key = f"{col}_std"
                if std_key in a1 and float(a1[std_key]) > 0:
                    out[f"{col}_z"] = out[col] / float(a1[std_key])
            except (TypeError, ValueError):
                pass
    return out


# ─────────────────────────────────────────────────────────────────────────────
# PART 2: conditional-on-trade return distribution
# ─────────────────────────────────────────────────────────────────────────────


def _load_parquets_dir(directory: str | Path, max_files: int = 50,
                       max_per_subdir: int = 25) -> pd.DataFrame:
    """Load tapes from a directory, optionally with regime subdirs."""
    p = Path(directory)
    sub_dirs = [d for d in p.iterdir() if d.is_dir()]
    parquet_paths: list[Path] = []
    if sub_dirs:
        for sd in sorted(sub_dirs):
            parquet_paths.extend(sorted(sd.glob("*.parquet"))[:max_per_subdir])
    else:
        parquet_paths = sorted(p.glob("*.parquet"))[:max_files]
        if not parquet_paths:
            parquet_paths = sorted(p.glob("*/*.parquet"))[:max_files]
    if max_files and len(parquet_paths) > max_files:
        # Even subsample across subdirs
        parquet_paths = parquet_paths[:max_files]

    frames = []
    for pq in parquet_paths:
        try:
            df = pd.read_parquet(pq, columns=["mid_return"])
            frames.append(df)
        except Exception as e:
            print(f"  WARN: skipping {pq}: {e}", file=sys.stderr)
    if not frames:
        raise FileNotFoundError(f"no parquets loaded from {directory}")
    return pd.concat(frames, ignore_index=True)


def conditional_return_stats(label: str, mid_return: pd.Series) -> dict:
    """frac_zero + stats over the nonzero subset."""
    arr = pd.to_numeric(mid_return, errors="coerce").dropna().values
    n = len(arr)
    if n == 0:
        return {"model": label}
    is_zero = arr == 0.0
    nonzero = arr[~is_zero]
    out = {
        "model": label,
        "n": n,
        "frac_zero": float(is_zero.mean()),
        "n_nonzero": int(len(nonzero)),
    }
    if len(nonzero) >= 10:
        out.update({
            "uncond_std": float(np.std(arr)),
            "uncond_kurt": float(scipy_stats.kurtosis(arr)),
            "cond_mean": float(np.mean(nonzero)),
            "cond_std": float(np.std(nonzero)),
            "cond_skew": float(scipy_stats.skew(nonzero)),
            "cond_kurt": float(scipy_stats.kurtosis(nonzero)),
            "cond_min": float(np.min(nonzero)),
            "cond_max": float(np.max(nonzero)),
            "cond_p01": float(np.percentile(nonzero, 1)),
            "cond_p99": float(np.percentile(nonzero, 99)),
        })
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────


def write_report(out_dir: str, agent_rows: pd.DataFrame,
                 delta_rows: pd.DataFrame, cond_rows: pd.DataFrame) -> None:
    os.makedirs(out_dir, exist_ok=True)
    agent_rows.to_csv(f"{out_dir}/agent_decomposition.csv", index=False)
    delta_rows.to_csv(f"{out_dir}/a2_minus_a1_deltas.csv", index=False)
    cond_rows.to_csv(f"{out_dir}/conditional_returns.csv", index=False)

    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("RANKING-MECHANISM DIAGNOSTIC")
    lines.append("=" * 80)
    lines.append("")

    # Part 1 — per-agent decomposition
    lines.append("─" * 80)
    lines.append("PART 1: PER-AGENT METRIC DECOMPOSITION")
    lines.append("─" * 80)
    for model in agent_rows["model"].unique():
        sub = agent_rows[agent_rows["model"] == model]
        lines.append(f"\n  Model: {model}")
        lines.append(f"  {'agent':12s} {'n_days':>8s} {'pnl_mean':>14s} "
                     f"{'pnl_std':>14s} {'fills_mean':>12s} {'fill_rate':>10s} "
                     f"{'inv_var':>12s} {'drawdown':>14s}")
        for _, r in sub.iterrows():
            line = f"  {r['agent']:12s} {int(r.get('n_days', 0)):>8d} "
            for col in METRIC_COLS:
                key = f"{col}_mean"
                v = r.get(key)
                if pd.isna(v):
                    line += f"{'-':>14s} " if col != "fill_rate" else f"{'-':>10s} "
                elif col == "fill_rate":
                    line += f"{v:>10.4f} "
                elif col == "n_fills":
                    line += f"{v:>12.1f} "
                elif col == "inventory_var":
                    line += f"{v:>12.3g} "
                else:
                    line += f"{v:>14.6g} "
            # add std for terminal_pnl in a second column
            std_pnl = r.get("terminal_pnl_std", float('nan'))
            # already in line via std column above? we only printed mean; insert std
            lines.append(line.rstrip())

    # Part 1b — A2 minus A1 deltas
    lines.append("\n" + "─" * 80)
    lines.append("PART 1b: A2_AS_OFI − A1_AS DELTAS (does A2 win on PnL, fills, or inventory?)")
    lines.append("─" * 80)
    if len(delta_rows) > 0:
        z_cols = [c for c in delta_rows.columns if c.endswith("_z")]
        lines.append(f"\n  {'model':12s} " + " ".join(f"{c:>20s}" for c in z_cols))
        for _, r in delta_rows.iterrows():
            line = f"  {r['model']:12s} "
            for c in z_cols:
                v = r.get(c)
                if pd.isna(v):
                    line += f"{'-':>20s} "
                else:
                    line += f"{float(v):>+20.3f} "
            lines.append(line.rstrip())
        lines.append("\n  Effect size (delta / std_A1) — values >|0.3| are large; <|0.1| are noise-level.")
        lines.append("  Look for the single dimension that has a consistent sign + large magnitude")
        lines.append("  across truth/real (positive A2−A1) but reverses or shrinks in synth where")
        lines.append("  the ranking flipped (v6/v7_b). That's the actual driver of the ranking gap.")

    # Part 2 — conditional return distribution
    lines.append("\n" + "─" * 80)
    lines.append("PART 2: CONDITIONAL-ON-TRADE RETURN DISTRIBUTION")
    lines.append("─" * 80)
    if len(cond_rows) > 0:
        lines.append(f"\n  {'model':12s} {'n':>10s} {'frac_zero':>10s} "
                     f"{'cond_std':>14s} {'cond_kurt':>10s} {'cond_skew':>10s} "
                     f"{'cond_p01':>14s} {'cond_p99':>14s}")
        for _, r in cond_rows.iterrows():
            line = f"  {r['model']:12s} "
            line += f"{int(r.get('n', 0)):>10d} "
            line += f"{r.get('frac_zero', float('nan')):>10.4f} "
            line += f"{r.get('cond_std', float('nan')):>14.6g} "
            line += f"{r.get('cond_kurt', float('nan')):>10.1f} "
            line += f"{r.get('cond_skew', float('nan')):>10.3f} "
            line += f"{r.get('cond_p01', float('nan')):>14.6g} "
            line += f"{r.get('cond_p99', float('nan')):>14.6g} "
            lines.append(line.rstrip())
        lines.append("\n  If v2 has frac_zero ≈ 0 and cond_std ≪ real_cond_std but still")
        lines.append("  achieves ρ_diff = +0.40, then the agent ranking does NOT depend on")
        lines.append("  preserving the zero-inflation structure of mid_return.")
        lines.append("  → v9's dequantization buys validation-suite realism, not ρ_diff.")

    lines.append("")
    lines.append("=" * 80)
    lines.append("DECISION RULES FOR v9 INTERPRETATION")
    lines.append("=" * 80)
    lines.append("""
  If Part 1 shows A2 wins on `n_fills` or `fill_rate` (not terminal_pnl) in
  truth+real, the A1/A2 gap is a fill-rate / inventory mechanism. v9 needs
  trade arrival + size structure right (where dequantization helps), but
  not necessarily heavy-tailed mid_return.

  If Part 1 shows A2 wins on `terminal_pnl` directly (with no consistent
  fill_rate / n_fills delta), the gap is a return-prediction mechanism.
  v9's value will hinge on whether the dequantized copula preserves
  cross-feature correlation that the AS+OFI agent needs.

  If Part 2 shows v2 has frac_zero ≈ 0 yet wins ρ_diff, v9's dequantization
  buys marginal-distribution realism (good for the validation suite) but
  the ranking outcome is decoupled from zero-inflation. Don't expect v9
  to MOVE ρ_diff just because frac_zero went from 0.19 to 0.80.
""")

    report_path = f"{out_dir}/summary.txt"
    Path(report_path).write_text("\n".join(lines))
    print(f"\nReport written to {report_path}")
    print(f"CSVs: agent_decomposition.csv, a2_minus_a1_deltas.csv, conditional_returns.csv")
    for ln in lines:
        print(ln)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metrics", nargs="*", default=[],
                   help="label:path pairs of metrics.csv files to decompose")
    p.add_argument("--synth-dirs", nargs="*", default=[],
                   help="label:dir pairs for conditional-return analysis")
    p.add_argument("--real-dir", default=None,
                   help="Optional: directory of real val parquets for the conditional analysis")
    p.add_argument("--out", default="results/ranking_mechanism")
    p.add_argument("--max-files", type=int, default=50,
                   help="Per-model parquet cap for Part 2")
    args = p.parse_args()

    # ── Part 1: per-agent decomposition ──
    metrics_paths = _parse_kv(args.metrics)
    all_agent_rows: list[pd.DataFrame] = []
    delta_rows: list[dict] = []
    for label, mpath in metrics_paths.items():
        if not Path(mpath).exists():
            print(f"  skip {label}: {mpath} not found", file=sys.stderr)
            continue
        print(f"Loading metrics for {label}: {mpath}")
        df = load_metrics_csv(mpath)
        agg = aggregate_per_agent(df)
        agg.insert(0, "model", label)
        all_agent_rows.append(agg)

        deltas = a2_minus_a1_deltas(agg)
        if deltas:
            row = {"model": label, **deltas}
            delta_rows.append(row)

    if not all_agent_rows:
        print("No metrics loaded. Skipping Part 1.")
        agent_df = pd.DataFrame()
        delta_df = pd.DataFrame()
    else:
        agent_df = pd.concat(all_agent_rows, ignore_index=True)
        delta_df = pd.DataFrame(delta_rows)

    # ── Part 2: conditional return distribution ──
    cond_rows: list[dict] = []
    if args.real_dir:
        if Path(args.real_dir).exists():
            print(f"Loading real from {args.real_dir} for conditional-return stats")
            try:
                real_mr = _load_parquets_dir(args.real_dir, max_files=args.max_files)
                cond_rows.append(conditional_return_stats("real", real_mr["mid_return"]))
            except Exception as e:
                print(f"  WARN: real-dir load failed: {e}", file=sys.stderr)

    synth_paths = _parse_kv(args.synth_dirs)
    for label, sdir in synth_paths.items():
        if not Path(sdir).exists():
            print(f"  skip {label}: {sdir} not found", file=sys.stderr)
            continue
        print(f"Loading synth for {label} from {sdir}")
        try:
            sdf = _load_parquets_dir(sdir, max_files=args.max_files)
            cond_rows.append(conditional_return_stats(label, sdf["mid_return"]))
        except Exception as e:
            print(f"  WARN: synth-dir load failed for {label}: {e}", file=sys.stderr)

    cond_df = pd.DataFrame(cond_rows)

    # Write report (handles empty inputs gracefully)
    write_report(args.out, agent_df, delta_df, cond_df)


if __name__ == "__main__":
    main()
