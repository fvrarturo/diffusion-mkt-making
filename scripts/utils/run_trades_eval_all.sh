#!/bin/bash
# Run BOTH script 62 (TRADES supplementary metrics) and script 63 (predictive
# score) over every generator we have synth tapes for. CPU only.
#
# Usage (in tmux so you can detach):
#   tmux new -s trades_eval
#   ./scripts/utils/run_trades_eval_all.sh \
#       2>&1 | tee logs/trades_eval_all_$(date +%Y%m%d_%H%M%S).log
#   # Ctrl-b d to detach
#
# Total wall time: ~30 min (script 62) + ~2-3h (script 63 trains 18+ models).
#
# ── Notes on robustness ────────────────────────────────────────────────────
# - `set -euo pipefail`: stop on first error, treat unset vars as errors,
#   propagate failures through pipes.
# - No nested tees: the user-level `tee` (in the invocation above) captures
#   everything. Adding a per-stage tee inside the script caused output loss
#   in tmux on the cluster (silent stage exits, empty logs).
# - Each python command is echoed BEFORE execution so debugging is easy
#   when something fails.

set -euo pipefail

SYNTH_BASE="$HOME/orcd/scratch/diffmm"

# (label, synth_dir) pairs. Filtered at runtime to skip nonexistent dirs.
declare -A MODELS=(
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

log() { echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p logs results/trades_metrics_all results/predictive_score_all

# ─── Filter to existing synth dirs ──────────────────────────────────────────
LABELS=()
DIRS=()
for label in "${!MODELS[@]}"; do
    d="${MODELS[$label]}"
    if [[ -d "$d" ]] && find "$d" -name '*.parquet' -print -quit 2>/dev/null | grep -q .; then
        LABELS+=("$label")
        DIRS+=("$d")
    else
        log "SKIP $label  (no parquets at $d)"
    fi
done

if [[ ${#LABELS[@]} -eq 0 ]]; then
    log "FATAL: no models found with parquets — nothing to do"
    exit 1
fi

# Sort labels alphabetically for deterministic ordering in plots
N=${#LABELS[@]}
ORDER_FILE=$(mktemp)
for i in $(seq 0 $((N - 1))); do
    echo "$i ${LABELS[$i]}"
done | sort -k2 > "$ORDER_FILE"
SORTED_LABELS=()
SORTED_DIRS=()
while read -r idx _; do
    SORTED_LABELS+=("${LABELS[$idx]}")
    SORTED_DIRS+=("${DIRS[$idx]}")
done < "$ORDER_FILE"
rm -f "$ORDER_FILE"
LABELS=("${SORTED_LABELS[@]}")
DIRS=("${SORTED_DIRS[@]}")

log "=== will evaluate ${#LABELS[@]} models ==="
for i in $(seq 0 $((${#LABELS[@]} - 1))); do
    log "  ${LABELS[$i]}  ->  ${DIRS[$i]}"
done

# ─── Stage 1: TRADES supplementary metrics (script 62) ──────────────────────
log "════════════════════════════════════════════════════════════════"
log "=== STAGE 1: scripts/62_trades_metrics.py over ${#LABELS[@]} models ==="
log "════════════════════════════════════════════════════════════════"
log "Expected wall time: ~15-30 min (CPU)"

# Echo the exact command before running it (so debugging is easy)
log "EXEC: python scripts/62_trades_metrics.py --real-dir data/processed/val --synth-dirs ${DIRS[*]} --labels ${LABELS[*]} --out results/trades_metrics_all --max-tapes 100"

python scripts/62_trades_metrics.py \
    --real-dir data/processed/val \
    --synth-dirs "${DIRS[@]}" \
    --labels "${LABELS[@]}" \
    --out results/trades_metrics_all \
    --max-tapes 100

log "=== STAGE 1 DONE: results in results/trades_metrics_all/ ==="

# ─── Stage 2: predictive score (script 63) ──────────────────────────────────
log "════════════════════════════════════════════════════════════════"
log "=== STAGE 2: scripts/63_predictive_score.py over ${#LABELS[@]} models ==="
log "════════════════════════════════════════════════════════════════"
log "Expected wall time: ~2-3h (sequential LSTM training, ~5-10 min each)"

if [[ ! -d "data/processed/train" ]]; then
    log "FATAL: data/processed/train not found; predictive_score baseline can't run"
    exit 1
fi

log "EXEC: python scripts/63_predictive_score.py --real-train-dir data/processed/train --real-test-dir data/processed/val --synth-dirs ${DIRS[*]} --labels ${LABELS[*]} --out results/predictive_score_all --window-size 64 --stride 8 --max-tapes-train 200 --max-tapes-test 10 --seed 42 --device cpu"

python scripts/63_predictive_score.py \
    --real-train-dir data/processed/train \
    --real-test-dir data/processed/val \
    --synth-dirs "${DIRS[@]}" \
    --labels "${LABELS[@]}" \
    --out results/predictive_score_all \
    --window-size 64 \
    --stride 8 \
    --max-tapes-train 200 \
    --max-tapes-test 10 \
    --seed 42 \
    --device cpu

log "=== STAGE 2 DONE: results in results/predictive_score_all/ ==="

# ─── Summary ────────────────────────────────────────────────────────────────
log "════════════════════════════════════════════════════════════════"
log "=== ALL DONE — final outputs ==="
log "════════════════════════════════════════════════════════════════"
log "TRADES metrics:"
log "  results/trades_metrics_all/trades_metrics_summary.txt"
log "  results/trades_metrics_all/{vol_volatility_correlation,leverage_effect,pca_coverage}.{pdf,png,csv}"
log "  results/trades_metrics_all/midprice_traces_<model>.{pdf,png}"
log ""
log "Predictive score:"
log "  results/predictive_score_all/predictive_score.{csv,pdf,png}"
log ""
log "To view the predictive score table:"
log "  column -t -s, results/predictive_score_all/predictive_score.csv"
