#!/bin/bash
# Run ph6_validate_cpu (G1 16-check pass/fail summary, parts B/C/D) on every
# generator we have synth tapes for. Parallel submission with per-model
# output dirs (so they don't race). Then collate into one comparison table.
#
# Usage:
#   tmux new -s validate_all
#   ./scripts/utils/validate_all_models.sh \
#       2>&1 | tee logs/validate_all_$(date +%Y%m%d_%H%M%S).log
#   # detach: Ctrl-b d
#
# Skips models that already have an archived G1 CSV.

set -u

SYNTH_BASE_OLD="$HOME/orcd/scratch/diffmm"   # synthetic_v*_stitched/INTC
SYNTH_BASE_NEW="$HOME/orcd/scratch/diffmm/synthetic"  # intc_v*/INTC and synthetic_v2_remapped/INTC

N_SYNTH=250

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# (model_label, synth_dir) pairs — both pre-Phase-E (stitched) and Phase-E+ (raw)
declare -A MODELS=(
    [v2]="$SYNTH_BASE_OLD/synthetic_v2_stitched/INTC"
    [v2_noclip]="$SYNTH_BASE_OLD/synthetic_v2_noclip_stitched/INTC"
    [v3_e9]="$SYNTH_BASE_OLD/synthetic_v3_e9_stitched/INTC"
    [v3_e9_noclip]="$SYNTH_BASE_OLD/synthetic_v3_e9_noclip_stitched/INTC"
    [v3_e19]="$SYNTH_BASE_OLD/synthetic_v3_e19_stitched/INTC"
    [v3p5]="$SYNTH_BASE_OLD/synthetic_v3p5_stitched/INTC"
    [v3p5_noclip]="$SYNTH_BASE_OLD/synthetic_v3p5_noclip_stitched/INTC"
    [v4]="$SYNTH_BASE_OLD/synthetic_v4_stitched/INTC"
    [v4_noclip]="$SYNTH_BASE_OLD/synthetic_v4_noclip_stitched/INTC"
    [v5]="$SYNTH_BASE_OLD/synthetic_v5_stitched/INTC"
    [v6]="$SYNTH_BASE_OLD/synthetic_v6_stitched/INTC"
    [v7]="$SYNTH_BASE_OLD/synthetic_v7_stitched/INTC"
    [v7_b]="$SYNTH_BASE_OLD/synthetic_v7_b_stitched/INTC"
    [v8]="$SYNTH_BASE_NEW/intc_v8/INTC"
    [v8_b]="$SYNTH_BASE_NEW/intc_v8_b/INTC"
    [v9]="$SYNTH_BASE_NEW/intc_v9/INTC"
    [v9_b]="$SYNTH_BASE_NEW/intc_v9_b/INTC"
    [v10_cascade]="$SYNTH_BASE_NEW/intc_v10_cascade/INTC"
    [v11]="$SYNTH_BASE_NEW/intc_v11/INTC"
    [v2_remapped]="$SYNTH_BASE_OLD/synthetic_v2_remapped/INTC"
)

mkdir -p results/_archive logs

# ─── Submit one validation per model (parallel, per-model OUT_DIR) ───
log "=== validating ${#MODELS[@]} models in parallel ==="
declare -A JIDS
declare -A OUT_DIRS
for model in "${!MODELS[@]}"; do
    synth_dir="${MODELS[$model]}"
    arch_dir="results/_archive/$model/validation"

    if [[ -f "$arch_dir/G1_pass_fail_summary.csv" ]]; then
        log "[$model] SKIP — archived G1 CSV already exists at $arch_dir"
        continue
    fi
    if [[ ! -d "$synth_dir" ]]; then
        log "[$model] SKIP — synth dir not found: $synth_dir"
        continue
    fi

    out_dir="results/validation_${model}/INTC"
    OUT_DIRS[$model]="$out_dir"
    log "[$model] submitting ph6_validate_cpu — synth=$synth_dir  out=$out_dir"
    jid=$(SYNTH_DIR="$synth_dir" N_SYNTH="$N_SYNTH" OUT_DIR="$out_dir" \
        sbatch --parsable slurms/ph6_validate_cpu.slurm 2>&1)
    if [[ "$jid" =~ ^[0-9]+$ ]]; then
        JIDS[$model]=$jid
        log "[$model] submitted as JID=$jid"
    else
        log "[$model] FAILED to submit: $jid"
    fi
done

if [[ ${#JIDS[@]} -eq 0 ]]; then
    log "Nothing to submit. Skipping wait + collate."
fi

# ─── Wait for all submitted jobs to drain ───
log "=== waiting for ${#JIDS[@]} validation jobs to finish ==="
while true; do
    running=0
    for model in "${!JIDS[@]}"; do
        if squeue -u favara -j "${JIDS[$model]}" -h 2>/dev/null | grep -q .; then
            running=$((running + 1))
        fi
    done
    if [[ $running -eq 0 ]]; then break; fi
    log "  $running / ${#JIDS[@]} validation jobs still running..."
    sleep 60
done
log "=== all validation jobs finished ==="

# ─── Archive each completed validation to results/_archive/<model>/validation ───
log "=== archiving validation outputs ==="
for model in "${!OUT_DIRS[@]}"; do
    out_dir="${OUT_DIRS[$model]}"
    arch_dir="results/_archive/$model/validation"
    if [[ -d "$out_dir" && -f "$out_dir/G1_pass_fail_summary.csv" ]]; then
        mkdir -p "$arch_dir"
        cp -r "$out_dir"/* "$arch_dir/"
        log "[$model] archived to $arch_dir ($(find $arch_dir -type f | wc -l) files)"
    else
        log "[$model] WARN: no G1 CSV at $out_dir — validation may have failed"
    fi
done

# ─── Collate G1 results into one comparison table ───
log "=== collating G1 across all models ==="
python <<'EOF'
import csv, os
from pathlib import Path

# Discover all archived validation dirs
arch_root = Path("results/_archive")
models_with_g1 = []
for d in sorted(arch_root.iterdir()):
    if not d.is_dir():
        continue
    g1 = d / "validation" / "G1_pass_fail_summary.csv"
    if g1.exists():
        models_with_g1.append((d.name, g1))

print(f"\nFound G1 CSVs for {len(models_with_g1)} models:")
for name, _ in models_with_g1:
    print(f"  - {name}")
print()

# Load each G1 — assume rows are (#, Check, Section, Value, Criterion, Pass)
all_data = {}      # model -> {check: pass_bool}
all_values = {}    # model -> {check: value_str}
check_order = []
for name, path in models_with_g1:
    rows = []
    with open(path, newline='') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 6:
                continue
            rows.append(row)
    all_data[name] = {}
    all_values[name] = {}
    for r in rows:
        idx, check, section, value, criterion, pass_str = r[0], r[1], r[2], r[3], r[4], r[5]
        # Strip non-ASCII bytes from check name to make it printable
        check_clean = check.encode('ascii', 'ignore').decode('ascii').strip()
        if check_clean not in check_order:
            check_order.append(check_clean)
        all_data[name][check_clean] = pass_str.strip()
        all_values[name][check_clean] = value.strip()

# Print pass/fail matrix (rows = checks, cols = models)
model_names = sorted(all_data.keys())
print(f"\n{'='*100}")
print(f"G1 PASS/FAIL MATRIX  ({len(check_order)} checks × {len(model_names)} models)")
print(f"{'='*100}\n")
# Header
hdr = f"{'check':<35s}  " + "  ".join(f"{m:>10s}" for m in model_names)
print(hdr)
print("-" * len(hdr))
pass_counts = {m: 0 for m in model_names}
total_counts = {m: 0 for m in model_names}
for check in check_order:
    line = f"{check[:34]:<35s}  "
    for m in model_names:
        v = all_data[m].get(check, "")
        if v == "True":
            mark = "PASS"
            pass_counts[m] += 1
            total_counts[m] += 1
        elif v == "False":
            mark = "fail"
            total_counts[m] += 1
        else:
            mark = "-"
        line += f"  {mark:>10s}"
    print(line)
print("-" * len(hdr))
print(f"{'TOTAL PASS':<35s}  " + "  ".join(
    f"{pass_counts[m]:>4d}/{total_counts[m]:<4d} " for m in model_names))

# Save to a CSV for easy import
out_csv = Path("results/_archive/G1_comparison_all_models.csv")
out_csv.parent.mkdir(parents=True, exist_ok=True)
with open(out_csv, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(["check"] + [f"{m}_pass" for m in model_names] + [f"{m}_value" for m in model_names])
    for check in check_order:
        row = [check]
        row += [all_data[m].get(check, "") for m in model_names]
        row += [all_values[m].get(check, "") for m in model_names]
        w.writerow(row)
    w.writerow(["TOTAL_PASS"] + [f"{pass_counts[m]}/{total_counts[m]}" for m in model_names] + [""] * len(model_names))
print(f"\nWrote comparison CSV to {out_csv}")
EOF

log "=== ALL DONE ==="
