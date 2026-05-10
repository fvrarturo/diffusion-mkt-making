#!/usr/bin/env python3
"""
Join all comparison artifacts into a single multi-axis CSV — one row per model,
columns are the headline metrics from G1, TRADES, predictive score, ρ_diff.

Run locally after pulling results from cluster:
    python scripts/utils/build_comparison_matrix.py

Outputs:
    results/_archive/multi_axis_comparison.csv     # the joined matrix
    results/_archive/multi_axis_comparison.md      # markdown table for direct paste
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
ARCHIVE = RESULTS / "_archive"


def load_g1_pass_counts() -> dict[str, tuple[int, int]]:
    """Read G1_comparison_all_models.csv → {model: (pass_count, total)}."""
    p = ARCHIVE / "G1_comparison_all_models.csv"
    if not p.exists():
        return {}
    out: dict[str, tuple[int, int]] = {}
    with open(p) as f:
        reader = csv.reader(f)
        header = next(reader)
        # Find columns that end with _pass
        pass_cols = [(i, h.removesuffix("_pass")) for i, h in enumerate(header) if h.endswith("_pass")]
        rows = list(reader)
        # Last row should be TOTAL_PASS
        total_row = next((r for r in rows if r and r[0].startswith("TOTAL")), None)
        if total_row:
            for i, label in pass_cols:
                if i < len(total_row):
                    cell = total_row[i]
                    m = re.match(r"(\d+)/(\d+)", cell)
                    if m:
                        out[label] = (int(m.group(1)), int(m.group(2)))
    return out


def load_predictive_scores() -> dict[str, dict]:
    """predictive_score.csv → {model: {test_mae, relative_to_replay}}."""
    p = RESULTS / "predictive_score_all" / "predictive_score.csv"
    if not p.exists():
        return {}
    out: dict[str, dict] = {}
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("label") or row.get("model")
            if not label:
                continue
            try:
                out[label] = {
                    "test_mae": float(row.get("test_mae", "nan")),
                    "predictive_x_replay": float(row.get("relative_to_replay", "nan")),
                }
            except ValueError:
                continue
    return out


def load_pca_coverage() -> dict[str, float]:
    """pca_coverage.csv → {model: coverage_pct}."""
    p = RESULTS / "trades_metrics_all" / "pca_coverage.csv"
    if not p.exists():
        return {}
    out: dict[str, float] = {}
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("model")
            if not label:
                continue
            try:
                out[label] = float(row.get("coverage_pct", "nan"))
            except ValueError:
                continue
    return out


def load_vol_volatility() -> dict[str, float]:
    """vol_volatility_correlation.csv → {model: mean per-tape Pearson}."""
    p = RESULTS / "trades_metrics_all" / "vol_volatility_correlation.csv"
    if not p.exists():
        return {}
    # The CSV is per-tape. Aggregate to mean per model.
    sums: dict[str, list[float]] = {}
    with open(p) as f:
        reader = csv.DictReader(f)
        for row in reader:
            label = row.get("model")
            try:
                v = float(row.get("pearson_r", "nan"))
                sums.setdefault(label, []).append(v)
            except (ValueError, TypeError):
                continue
    return {m: sum(vs) / len(vs) for m, vs in sums.items() if vs}


def load_rho_diff() -> dict[str, dict]:
    """Per-model bootstrap.json → {model: {rho_diff, pi_diff}}."""
    out: dict[str, dict] = {}
    for d in ARCHIVE.iterdir():
        if not d.is_dir():
            continue
        bs = d / "hypothesis" / "bootstrap.json"
        if not bs.exists():
            continue
        try:
            data = json.loads(bs.read_text())
            out[d.name] = {
                "rho_diff": float(data.get("rho_diff_point", float("nan"))),
                "rho_hist": float(data.get("rho_hist_point", float("nan"))),
                "pi_diff": ",".join(data.get("pi_diff", [])),
                "p_value_one_sided": float(data.get("p_value_one_sided", float("nan"))),
            }
        except (json.JSONDecodeError, ValueError):
            continue
    return out


def _read_csv_column(path: Path, col: str, key_col: str = None, key_val: str = None) -> str | float | None:
    """Read a single value from a CSV. If key_col/key_val are set, find the row
    where key_col == key_val. Otherwise return the first non-header row's `col`."""
    if not path.exists():
        return None
    try:
        with open(path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                if key_col and row.get(key_col) != key_val:
                    continue
                v = row.get(col)
                if v is None or v == "":
                    return None
                try:
                    return float(v)
                except (ValueError, TypeError):
                    return v
    except Exception:
        return None
    return None


def load_b1_synth_stats() -> dict[str, dict]:
    """Per-model B1_return_stats.csv → synth row {std, kurt, frac_zero}.
    Schema: key, mean, std, skew, kurt, min, max, p1, p5, p95, p99, n, val
    Rows we want: 'synth' row gives these stats; 'wasserstein_1' / 'pass_*' rows
    are derived metrics in the `val` column.
    """
    out: dict[str, dict] = {}
    for d in ARCHIVE.iterdir():
        if not d.is_dir():
            continue
        p = d / "validation" / "B1_return_stats.csv"
        if not p.exists():
            continue
        try:
            with open(p) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            synth_row = next((r for r in rows if r.get("key") == "synth"), None)
            kurt_row = next((r for r in rows if r.get("key") == "pass_kurt"), None)
            w1_row = next((r for r in rows if r.get("key") == "wasserstein_1"), None)
            if synth_row:
                m = {}
                for col in ("std", "skew", "kurt"):
                    v = synth_row.get(col)
                    try: m[f"B1_synth_{col}"] = float(v) if v else None
                    except: m[f"B1_synth_{col}"] = None
                # frac_zero isn't a column; derive from p5==0 / etc. — actually
                # the schema doesn't expose it directly. Skip for this CSV.
                out[d.name] = m
                if w1_row and w1_row.get("val"):
                    try: out[d.name]["B1_w1"] = float(w1_row["val"])
                    except: pass
        except Exception:
            continue
    return out


def load_d1_ofi_slopes() -> dict[str, dict]:
    """Per-model D1_regression_stats.csv → {model: {ofi_slope_d50, ofi_slope_sign_d50}}.
    Schema: delta, beta_real, beta_synth, r2_real, r2_synth, pass_sign
    """
    out: dict[str, dict] = {}
    for d in ARCHIVE.iterdir():
        if not d.is_dir():
            continue
        p = d / "validation" / "D1_regression_stats.csv"
        if not p.exists():
            continue
        try:
            with open(p) as f:
                reader = csv.DictReader(f)
                rows = list(reader)
            row50 = next((r for r in rows if r.get("delta") == "50"), None)
            row10 = next((r for r in rows if r.get("delta") == "10"), None)
            m = {}
            if row50:
                try:
                    m["D1_ofi_slope_d50"] = float(row50.get("beta_synth", "nan"))
                except (ValueError, TypeError):
                    pass
            if row10:
                try:
                    m["D1_ofi_slope_d10"] = float(row10.get("beta_synth", "nan"))
                except (ValueError, TypeError):
                    pass
            if m:
                out[d.name] = m
        except Exception:
            continue
    return out


def load_b4_trade_fraction() -> dict[str, float]:
    """Per-model B4_event_stats.csv → trade fraction.
    Schema is metric,value with rows including 'synth_trade_frac'.
    """
    out: dict[str, float] = {}
    for d in ARCHIVE.iterdir():
        if not d.is_dir():
            continue
        p = d / "validation" / "B4_event_stats.csv"
        if not p.exists():
            continue
        try:
            with open(p) as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get("metric") == "synth_trade_frac":
                        try:
                            out[d.name] = float(row.get("value", "nan"))
                        except ValueError:
                            pass
                        break
        except Exception:
            continue
    return out


def load_real_reference() -> dict:
    """Pull real-side reference values from any one model's per-check CSVs.
    Real values are identical across models (they're computed from the same val
    set), so we just grab from the first one we find."""
    ref = {}
    # B1 — real row of B1_return_stats.csv
    for d in ARCHIVE.iterdir():
        if not d.is_dir(): continue
        p = d / "validation" / "B1_return_stats.csv"
        if not p.exists(): continue
        try:
            with open(p) as f:
                rows = list(csv.DictReader(f))
            real_row = next((r for r in rows if r.get("key") == "real"), None)
            if real_row:
                for col in ("std", "skew", "kurt"):
                    v = real_row.get(col)
                    try:
                        ref[f"B1_synth_{col}"] = float(v) if v else None
                    except ValueError:
                        pass
                # inter-day W-1 is the real-baseline criterion
                idw1_row = next((r for r in rows if r.get("key") == "inter_day_w1_mean"), None)
                if idw1_row and idw1_row.get("val"):
                    try:
                        ref["B1_w1"] = float(idw1_row["val"])  # real reference for W-1
                    except ValueError:
                        pass
                break
        except Exception:
            continue
    # D1 — beta_real columns
    for d in ARCHIVE.iterdir():
        if not d.is_dir(): continue
        p = d / "validation" / "D1_regression_stats.csv"
        if not p.exists(): continue
        try:
            with open(p) as f:
                rows = list(csv.DictReader(f))
            row10 = next((r for r in rows if r.get("delta") == "10"), None)
            row50 = next((r for r in rows if r.get("delta") == "50"), None)
            if row10 and row10.get("beta_real"):
                try: ref["D1_ofi_slope_d10"] = float(row10["beta_real"])
                except ValueError: pass
            if row50 and row50.get("beta_real"):
                try: ref["D1_ofi_slope_d50"] = float(row50["beta_real"])
                except ValueError: pass
            break
        except Exception:
            continue
    # B4 — real_trade_frac
    for d in ARCHIVE.iterdir():
        if not d.is_dir(): continue
        p = d / "validation" / "B4_event_stats.csv"
        if not p.exists(): continue
        try:
            with open(p) as f:
                for row in csv.DictReader(f):
                    if row.get("metric") == "real_trade_frac":
                        try:
                            ref["B4_trade_fraction"] = float(row.get("value", "nan"))
                        except ValueError:
                            pass
                        break
            break
        except Exception:
            continue
    return ref


def load_window_degenerate() -> dict[str, float]:
    """From deep_diagnostic/window_level_stats.csv if present."""
    p = RESULTS / "deep_diagnostic" / "window_level_stats.csv"
    if not p.exists():
        return {}
    out: dict[str, list[bool]] = {}
    try:
        with open(p) as f:
            reader = csv.DictReader(f)
            for row in reader:
                m = row.get("model")
                if not m:
                    continue
                deg = row.get("is_degenerate", "").lower() in ("true", "1")
                out.setdefault(m, []).append(deg)
    except Exception:
        return {}
    return {m: sum(vals) / len(vals) for m, vals in out.items() if vals}


def main():
    g1 = load_g1_pass_counts()
    pred = load_predictive_scores()
    pca = load_pca_coverage()
    vv = load_vol_volatility()
    rho = load_rho_diff()
    b1 = load_b1_synth_stats()
    d1 = load_d1_ofi_slopes()
    b4 = load_b4_trade_fraction()
    win_deg = load_window_degenerate()
    real_ref = load_real_reference()

    # Universe of models = union of all sources
    all_models = sorted(set(g1) | set(pred) | set(pca) | set(vv) | set(rho) | set(b1) | set(d1) | set(b4))

    rows = []

    # ── Real reference row at the top ──
    real_row = {
        "model": "REAL",
        "G1_pass": "16/16",            # by definition real passes everything
        "rho_diff": "",
        "rho_hist": +0.40,             # ρ_hist between val and holdout
        "pi_diff": "",
        "p_value_one_sided": "",
        "predictive_x_replay": +1.00,  # market replay baseline = real_train → real_val
        "predictive_test_mae": 1.1496e-04,
        "PCA_coverage_pct": 100.0,     # by definition (the hull is built from real)
        "vol_volatility_corr": vv.get("real", ""),
        "B1_synth_std": real_ref.get("B1_synth_std", ""),
        "B1_synth_skew": real_ref.get("B1_synth_skew", ""),
        "B1_synth_kurt": real_ref.get("B1_synth_kurt", ""),
        "B1_w1": real_ref.get("B1_w1", ""),
        "D1_ofi_slope_d10": real_ref.get("D1_ofi_slope_d10", ""),
        "D1_ofi_slope_d50": real_ref.get("D1_ofi_slope_d50", ""),
        "B4_trade_fraction": real_ref.get("B4_trade_fraction", ""),
        "frac_degenerate_windows": 0.0,
    }
    rows.append(real_row)

    for m in all_models:
        if m in ("real", "REAL", "market_replay"):
            # market_replay is a baseline, not a generator — keep it but flag
            if m == "market_replay":
                rows.append({
                    "model": "market_replay",
                    "G1_pass": "—",
                    "rho_diff": "", "rho_hist": "", "pi_diff": "", "p_value_one_sided": "",
                    "predictive_x_replay": +1.00,
                    "predictive_test_mae": pred.get(m, {}).get("test_mae", ""),
                    "PCA_coverage_pct": "", "vol_volatility_corr": "",
                    "B1_synth_std": "", "B1_synth_skew": "", "B1_synth_kurt": "",
                    "B1_w1": "", "D1_ofi_slope_d10": "", "D1_ofi_slope_d50": "",
                    "B4_trade_fraction": "", "frac_degenerate_windows": "",
                })
            continue
        g = g1.get(m)
        p = pred.get(m, {})
        r = rho.get(m, {})
        b = b1.get(m, {})
        d_ = d1.get(m, {})
        rows.append({
            "model": m,
            "G1_pass": f"{g[0]}/{g[1]}" if g else "-",
            "rho_diff": r.get("rho_diff", ""),
            "rho_hist": r.get("rho_hist", ""),
            "pi_diff": r.get("pi_diff", ""),
            "p_value_one_sided": r.get("p_value_one_sided", ""),
            "predictive_x_replay": p.get("predictive_x_replay", ""),
            "predictive_test_mae": p.get("test_mae", ""),
            "PCA_coverage_pct": pca.get(m, ""),
            "vol_volatility_corr": vv.get(m, ""),
            "B1_synth_std": b.get("B1_synth_std", ""),
            "B1_synth_skew": b.get("B1_synth_skew", ""),
            "B1_synth_kurt": b.get("B1_synth_kurt", ""),
            "B1_w1": b.get("B1_w1", ""),
            "D1_ofi_slope_d10": d_.get("D1_ofi_slope_d10", ""),
            "D1_ofi_slope_d50": d_.get("D1_ofi_slope_d50", ""),
            "B4_trade_fraction": b4.get(m, ""),
            "frac_degenerate_windows": win_deg.get(m, ""),
        })

    cols = ["model", "G1_pass", "rho_diff", "rho_hist", "pi_diff", "p_value_one_sided",
            "predictive_x_replay", "predictive_test_mae", "PCA_coverage_pct", "vol_volatility_corr",
            "B1_synth_std", "B1_synth_skew", "B1_synth_kurt", "B1_w1",
            "D1_ofi_slope_d10", "D1_ofi_slope_d50",
            "B4_trade_fraction", "frac_degenerate_windows"]
    out_csv = ARCHIVE / "multi_axis_comparison.csv"
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"Wrote {out_csv}")

    # Markdown table for direct paste into reports/lab notes
    md_path = ARCHIVE / "multi_axis_comparison.md"
    md_lines = []
    md_lines.append("# Multi-axis comparison (auto-generated)\n")
    md_lines.append("Real reference values: kurt~755, OFI slope at Δ=50 ≈ +1.11e-04, vol-vol corr ≈ +0.341\n")
    md_lines.append("| Model | G1 | ρ_diff | Predict×replay | PCA% | VolVol | B1_kurt | B1_std | OFI Δ=10 | OFI Δ=50 | trade frac |")
    md_lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    def fmt(x, sd=3):
        if x == "" or x is None: return "—"
        try:
            f = float(x)
            if abs(f) < 1e-3 or abs(f) >= 1e4: return f"{f:+.2e}"
            return f"{f:+.{sd}g}"
        except (ValueError, TypeError):
            return str(x)[:32]
    for r in rows:
        md_lines.append(
            f"| {r['model']} | {r['G1_pass']} | {fmt(r['rho_diff'])} | "
            f"{fmt(r['predictive_x_replay'], 3)} | {fmt(r['PCA_coverage_pct'], 3)} | "
            f"{fmt(r['vol_volatility_corr'], 3)} | {fmt(r['B1_synth_kurt'], 3)} | "
            f"{fmt(r['B1_synth_std'])} | {fmt(r['D1_ofi_slope_d10'])} | "
            f"{fmt(r['D1_ofi_slope_d50'])} | {fmt(r['B4_trade_fraction'], 3)} |"
        )
    md_path.write_text("\n".join(md_lines))
    print(f"Wrote {md_path}")
    print()
    print("=" * 100)
    for line in md_lines:
        print(line)
    print()
    print(f"=== {len(rows)} models in matrix ===")


if __name__ == "__main__":
    main()
