#!/bin/bash
# P0 — fill the three blocking analysis gaps before report writing:
#   B. deep_diagnostic for v5, v9, v9_b   (the "protagonists" §§4-6 missed)
#   C. conditional_returns for all 20 models (Table 5.1, §3.3)
#   D. per-regime G1 for v2, v5, v9       (§3.2.2 conditioning evidence)
#
# All three are CPU-light. Designed to run on the cluster login node in tmux:
#
#   tmux new -s p0
#   ./scripts/utils/run_p0_post_overnight.sh \
#       2>&1 | tee logs/p0_$(date +%Y%m%d_%H%M%S).log
#   # Ctrl-b d to detach
#
# Total wall time: ~30-45 min (sequential).
#
# Outputs (all under results/):
#   deep_diagnostic_v5_v9/{per_feature_stats, correlation_comparison,
#                          temporal_acf, window_level_stats, z_space_stats,
#                          regime_conditional, ofi_regression}.csv
#   ranking_mechanism_full/{conditional_returns, agent_decomposition,
#                           a2_minus_a1_deltas}.csv
#   _archive/<model>/validation_<regime>/   (per-regime B/C/D outputs)

set -uo pipefail

SYNTH_BASE="$HOME/orcd/scratch/diffmm"
COPULA_PATH="data/processed/norm_stats_INTC.copula.json"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# Helper: submit a python invocation as an sbatch job and wait for it.
# Usage:  submit_and_wait <name> <mem> <hours> -- python script.py args...
# Logs land in logs/<name>_<jid>.out / .err.
submit_and_wait() {
    local name="$1" mem="$2" hours="$3"
    shift 3
    [[ "$1" == "--" ]] && shift
    local cmd="$*"
    log "  EXEC (sbatch): $cmd"
    local jid
    jid=$(sbatch --parsable \
        --partition=mit_preemptable \
        --job-name="$name" \
        --cpus-per-task=4 \
        --mem="$mem" \
        --time="${hours}:00:00" \
        --output="logs/${name}_%j.out" \
        --error="logs/${name}_%j.err" \
        --wrap="source slurms/_common.sh 2>/dev/null || true; cd $PWD && $cmd")
    if [[ -z "$jid" ]]; then
        log "  ERROR: sbatch returned empty JID for $name"
        return 1
    fi
    log "  $name JID=$jid (waiting...)"
    while squeue -u "$USER" -j "$jid" -h 2>/dev/null | grep -q .; do
        sleep 60
    done
    # Print the tail of the job log
    if [[ -f "logs/${name}_${jid}.out" ]]; then
        log "  --- tail logs/${name}_${jid}.out ---"
        tail -10 "logs/${name}_${jid}.out" | sed 's/^/      /'
    fi
    if [[ -f "logs/${name}_${jid}.err" && -s "logs/${name}_${jid}.err" ]]; then
        log "  --- tail logs/${name}_${jid}.err ---"
        tail -10 "logs/${name}_${jid}.err" | sed 's/^/      /'
    fi
}

mkdir -p logs results

# ─── Build the (label, dir) mapping for all 20 INTC models ────────────────
declare -A SYNTH_DIR=(
    [v2]="$SYNTH_BASE/synthetic_v2_stitched/INTC"
    [v2_noclip]="$SYNTH_BASE/synthetic_v2_noclip_stitched/INTC"
    [v2_remapped]="$SYNTH_BASE/synthetic_v2_remapped/INTC"
    [v3_e9]="$SYNTH_BASE/synthetic_v3_e9_stitched/INTC"
    [v3_e9_noclip]="$SYNTH_BASE/synthetic_v3_e9_noclip_stitched/INTC"
    [v3_e19]="$SYNTH_BASE/synthetic_v3_e19_stitched/INTC"
    [v3p5]="$SYNTH_BASE/synthetic_v3p5_stitched/INTC"
    [v3p5_noclip]="$SYNTH_BASE/synthetic_v3p5_noclip_stitched/INTC"
    [v4]="$SYNTH_BASE/synthetic_v4_stitched/INTC"
    [v4_noclip]="$SYNTH_BASE/synthetic_v4_noclip_stitched/INTC"
    [v5]="$SYNTH_BASE/synthetic_v5_stitched/INTC"
    [v6]="$SYNTH_BASE/synthetic_v6_stitched/INTC"
    [v7]="$SYNTH_BASE/synthetic_v7_stitched/INTC"
    [v7_b]="$SYNTH_BASE/synthetic_v7_b_stitched/INTC"
    [v8]="$SYNTH_BASE/synthetic/intc_v8/INTC"
    [v8_b]="$SYNTH_BASE/synthetic/intc_v8_b/INTC"
    [v9]="$SYNTH_BASE/synthetic/intc_v9/INTC"
    [v9_b]="$SYNTH_BASE/synthetic/intc_v9_b/INTC"
    [v10_cascade]="$SYNTH_BASE/synthetic/intc_v10_cascade/INTC"
    [v11]="$SYNTH_BASE/synthetic/intc_v11/INTC"
)

# ───────────────────────────────────────────────────────────────────────────
# P0.B — deep diagnostic on v5, v9, v9_b
# ───────────────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════════════════════════"
log "=== P0.B — deep diagnostic on v5, v9, v9_b ==="
log "════════════════════════════════════════════════════════════════"

P0B_LABELS=(v5 v9 v9_b)
P0B_DIRS=()
for lab in "${P0B_LABELS[@]}"; do
    d="${SYNTH_DIR[$lab]:-}"
    if [[ -z "$d" || ! -d "$d" ]]; then
        log "WARN: $lab synth dir missing ($d) — skipping"
        continue
    fi
    P0B_DIRS+=("$d")
done

if [[ ${#P0B_DIRS[@]} -gt 0 ]]; then
    extra=""
    if [[ -f "$COPULA_PATH" ]]; then
        extra="--copula-json $COPULA_PATH --copula-models v9 v9_b"
    fi
    submit_and_wait p0b 128G 2 -- \
        python scripts/60_deep_diagnostic.py \
            --real-dir data/processed/val \
            --synth-dirs "${P0B_DIRS[@]}" \
            --labels "${P0B_LABELS[@]}" \
            --out results/deep_diagnostic_v5_v9 \
            --max-tapes 50 \
            $extra \
        || log "WARN: P0.B sbatch wrapper failed"
    log "P0.B done — results in results/deep_diagnostic_v5_v9/"
else
    log "P0.B skipped (no synth dirs)"
fi

# ───────────────────────────────────────────────────────────────────────────
# P0.C — conditional_returns + agent decomposition for all 20 models
# ───────────────────────────────────────────────────────────────────────────
log ""
log "════════════════════════════════════════════════════════════════"
log "=== P0.C — conditional return stats for ALL 20 models ==="
log "════════════════════════════════════════════════════════════════"

P0C_LABELS=()
P0C_DIRS=()
for lab in "${!SYNTH_DIR[@]}"; do
    d="${SYNTH_DIR[$lab]}"
    if [[ -d "$d" ]] && find "$d" -name '*.parquet' -print -quit 2>/dev/null | grep -q .; then
        P0C_LABELS+=("$lab")
        P0C_DIRS+=("$d")
    fi
done

# Build the synth-dirs argument as colon-separated label:dir pairs (script 61's format)
# AND the metrics-csv arguments if backtest CSVs exist for the model.
SYNTH_ARGS=()
for i in "${!P0C_LABELS[@]}"; do
    SYNTH_ARGS+=("${P0C_LABELS[$i]}:${P0C_DIRS[$i]}")
done

METRICS_ARGS=()
# Pull metrics from each model's archived backtests if present
for lab in "${P0C_LABELS[@]}"; do
    m="results/_archive/$lab/synthetic_backtests/metrics.csv"
    if [[ -f "$m" ]]; then
        METRICS_ARGS+=("$lab:$m")
    fi
done
# Truth metrics (for the truth-row in agent_decomposition)
TRUTH_METRICS="results/_archive/v9/holdout/metrics.csv"
[[ -f "$TRUTH_METRICS" ]] && METRICS_ARGS=("truth:$TRUTH_METRICS" "${METRICS_ARGS[@]}")

METRICS_FLAG=""
if [[ ${#METRICS_ARGS[@]} -gt 0 ]]; then
    METRICS_FLAG="--metrics ${METRICS_ARGS[*]}"
fi

submit_and_wait p0c 128G 2 -- \
    python scripts/61_ranking_mechanism.py \
        $METRICS_FLAG \
        --synth-dirs "${SYNTH_ARGS[@]}" \
        --real-dir data/processed/val \
        --out results/ranking_mechanism_full \
        --max-files 30 \
    || log "WARN: P0.C sbatch wrapper failed"

log "P0.C done — results in results/ranking_mechanism_full/"

# ───────────────────────────────────────────────────────────────────────────
# P0.D — per-regime G1 for v2, v5, v9
# ───────────────────────────────────────────────────────────────────────────
log ""
log "════════════════════════════════════════════════════════════════"
log "=== P0.D — per-regime G1 for v2, v5, v9 ==="
log "════════════════════════════════════════════════════════════════"

P0D_MODELS=(v2 v5 v9)
REGIMES=(base high_vol toxic thin)

for model in "${P0D_MODELS[@]}"; do
    base_dir="${SYNTH_DIR[$model]:-}"
    if [[ -z "$base_dir" || ! -d "$base_dir" ]]; then
        log "WARN: $model synth dir missing — skipping"
        continue
    fi
    for regime in "${REGIMES[@]}"; do
        rdir="$base_dir/$regime"
        if [[ ! -d "$rdir" ]]; then
            log "  [$model/$regime] no regime dir at $rdir — skipping"
            continue
        fi
        n=$(find "$rdir" -name '*.parquet' | wc -l | tr -d ' ')
        if [[ $n -lt 5 ]]; then
            log "  [$model/$regime] only $n parquets — skipping"
            continue
        fi
        out_dir="results/_archive/$model/validation_${regime}"
        mkdir -p "$out_dir"
        # Pass the PARENT (base_dir) + --regime so script 50 finds the right
        # subdir. Wrapped in sbatch so it doesn't OOM the login node.
        submit_and_wait "p0d_${model}_${regime}" 64G 1 -- \
            python scripts/50_validate_generator.py \
                --parts B C D \
                --real-dir data/processed/val \
                --synth-dir "$base_dir" \
                --regime "$regime" \
                --ticker INTC \
                --out-dir "$out_dir" \
                --n-synth-per-regime 60 \
            || log "  WARN: P0.D [$model/$regime] sbatch wrapper failed"
    done
done

log "P0.D done — per-regime outputs under results/_archive/<model>/validation_<regime>/"

# ───────────────────────────────────────────────────────────────────────────
# Summary
# ───────────────────────────────────────────────────────────────────────────
log ""
log "════════════════════════════════════════════════════════════════"
log "=== P0 ALL DONE — pull these back to Mac ==="
log "════════════════════════════════════════════════════════════════"
log "  results/deep_diagnostic_v5_v9/"
log "  results/ranking_mechanism_full/"
log "  results/_archive/{v2,v5,v9}/validation_{base,high_vol,toxic,thin}/"
log ""
log "Then locally: python scripts/80_report_figures.py"
log "(figure script auto-merges new CSVs into the report figure suite)"
