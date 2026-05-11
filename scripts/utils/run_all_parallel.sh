#!/bin/bash
# Massively parallel cluster runner — submits every remaining analysis to
# mit_preemptable as independent sbatch jobs, then waits once for all of
# them to drain. Replaces the serial submit_and_wait pattern of run_p0_*
# with a fan-out-everything-at-once pattern that takes advantage of the
# preemptable partition's >50-node fan-out (proven during SPY clean recovery).
#
# Designed to be IDEMPOTENT — each job has a "skip if output already
# exists" guard, so re-running picks up only what's missing. Safe to run
# while the old serial p0 orchestrator is still alive (the running p0d_v5_*
# sbatch jobs will skip themselves here when their outputs land).
#
# Usage:
#   tmux new -s parallel
#   ./scripts/utils/run_all_parallel.sh \
#       2>&1 | tee logs/parallel_$(date +%Y%m%d_%H%M%S).log
#   # Ctrl-b d
#
# Total wall time on preemptable (no priority queue): ~10-25 min for the
# whole batch — every job runs concurrently. Pull results with the
# rsync block at the end of this file.

set -uo pipefail

SYNTH_BASE="$HOME/orcd/scratch/diffmm"
CKPT_ROOT="$HOME/orcd/scratch/diffmm/checkpoints"
COPULA_PATH="data/processed/norm_stats_INTC.copula.json"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

mkdir -p logs results results/trades_style results/microstructure_diagnostics

# ─── Synth dir mapping (same as P0 orchestrator) ──────────────────────────
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
ALL_MODELS=(v2 v2_noclip v2_remapped v3_e9 v3_e9_noclip v3_e19 v3p5 v3p5_noclip
            v4 v4_noclip v5 v6 v7 v7_b v8 v8_b v9 v9_b v10_cascade v11)

# ─── Helper: submit one preemptable job, return JID ───────────────────────
submit_one() {
    local name="$1" mem="$2" hours="$3" extra_flags="${4:-}"
    shift 4
    local cmd="$*"
    local jid
    jid=$(sbatch --parsable \
        --partition=mit_preemptable \
        --job-name="$name" \
        --cpus-per-task=4 \
        --mem="$mem" \
        --time="${hours}:00:00" \
        --output="logs/${name}_%j.out" \
        --error="logs/${name}_%j.err" \
        $extra_flags \
        --wrap="source slurms/_common.sh 2>/dev/null || true; cd $PWD && $cmd" 2>/dev/null)
    if [[ -z "$jid" ]]; then
        log "  ERROR: sbatch failed for $name"
        return 1
    fi
    log "  → submitted $name  JID=$jid"
    echo "$jid"
}

declare -a JIDS
declare -A NAMES   # JID → human name

# ══════════════════════════════════════════════════════════════════════════
# JOB CATALOG — every analysis we want to run, with skip-if-exists guards
# ══════════════════════════════════════════════════════════════════════════

# ─── P0.D remaining (per-regime G1 for v5 and v9) ────────────────────────
log "════════════════════════════════════════════════════════════════"
log "Cataloguing jobs..."
log "════════════════════════════════════════════════════════════════"
log ""
log "── P0.D — per-regime G1 (v5, v9; v2 already done) ──"
for model in v5 v9; do
    base_dir="${SYNTH_DIR[$model]}"
    for regime in base high_vol toxic thin; do
        out_dir="results/_archive/$model/validation_${regime}"
        if [[ -f "$out_dir/G1_pass_fail_summary.csv" ]]; then
            log "  SKIP $model/$regime (output exists)"
            continue
        fi
        mkdir -p "$out_dir"
        cmd="python scripts/50_validate_generator.py --parts B C D \
              --real-dir data/processed/val \
              --synth-dir '$base_dir' --regime $regime \
              --ticker INTC --out-dir '$out_dir' --n-synth-per-regime 60"
        jid=$(submit_one "p0d_${model}_${regime}" 64G 1 "" "$cmd") && {
            JIDS+=("$jid"); NAMES[$jid]="p0d_${model}_${regime}"
        }
    done
done

# ─── P1.A — re-run script 62 with pca_projections.csv now written ─────────
log ""
log "── P1.A — pca_projections re-run (writes pca_projections.csv) ──"
if [[ ! -f results/trades_metrics_all/pca_projections.csv ]]; then
    # Filter to existing synth dirs, build flat label/dir lists
    LABELS=()
    DIRS=()
    for lab in "${ALL_MODELS[@]}"; do
        d="${SYNTH_DIR[$lab]}"
        if [[ -d "$d" ]] && find "$d" -name '*.parquet' -print -quit 2>/dev/null | grep -q .; then
            LABELS+=("$lab"); DIRS+=("$d")
        fi
    done
    cmd="python scripts/62_trades_metrics.py --real-dir data/processed/val \
          --synth-dirs ${DIRS[*]} --labels ${LABELS[*]} \
          --out results/trades_metrics_all --max-tapes 60"
    jid=$(submit_one p1a_trades_metrics 128G 2 "" "$cmd") && {
        JIDS+=("$jid"); NAMES[$jid]="p1a_trades_metrics"
    }
else
    log "  SKIP P1.A (pca_projections.csv already exists)"
fi

# ─── P1.D — real stylized facts (standalone real-INTC panel) ──────────────
log ""
log "── P1.D — real stylized facts (script 81) ──"
if [[ ! -f results/report_figures/fig_a_real_stylized_facts.pdf ]]; then
    cmd="python scripts/81_real_stylized_facts.py --real-dir data/processed/val --ticker INTC --max-files 10"
    jid=$(submit_one p1d_real_stylized 64G 1 "" "$cmd") && {
        JIDS+=("$jid"); NAMES[$jid]="p1d_real_stylized"
    }
else
    log "  SKIP P1.D (output exists)"
fi

# ─── P2 — microstructure diagnostics (spread + transitions + intraday) ────
log ""
log "── P2 — microstructure trio (script 83) ──"
if [[ ! -f results/microstructure_diagnostics/spread_conditional.csv ]]; then
    cmd="python scripts/83_microstructure_diagnostics.py \
          --real-dir data/processed/val \
          --synth-base $SYNTH_BASE \
          --models v2 v5 v9 \
          --out results/microstructure_diagnostics"
    jid=$(submit_one p2_microstructure 64G 1 "" "$cmd") && {
        JIDS+=("$jid"); NAMES[$jid]="p2_microstructure"
    }
else
    log "  SKIP P2 microstructure (output exists)"
fi

# ─── P2.G3 — regime t-SNE (light, but easy to ship to preemptable) ────────
log ""
log "── P2.G3 — regime embedding t-SNE (script 84) ──"
if [[ ! -f results/report_figures/fig_p2_4_regime_tsne.pdf ]]; then
    CKPT_ARGS=()
    for model in v2 v5 v9; do
        d="$CKPT_ROOT/intc_$model"
        c=$(ls -1 "$d"/ckpt-epoch*-valloss*.ckpt 2>/dev/null \
              | awk -F'valloss' '{print $2, $0}' \
              | sort -n | head -1 | awk '{print $2}')
        [[ -n "$c" ]] && CKPT_ARGS+=("$c:$model")
    done
    if [[ ${#CKPT_ARGS[@]} -gt 0 ]]; then
        cmd="python scripts/84_regime_embedding_tsne.py --ckpts ${CKPT_ARGS[*]}"
        jid=$(submit_one p2g3_tsne 16G 1 "" "$cmd") && {
            JIDS+=("$jid"); NAMES[$jid]="p2g3_tsne"
        }
    else
        log "  SKIP P2.G3 (no checkpoints found)"
    fi
else
    log "  SKIP P2.G3 (output exists)"
fi

# ─── NEW (TRADES-style) — intraday LOB volume (script 86) ─────────────────
log ""
log "── NEW — intraday LOB volume (script 86) ──"
if [[ ! -f results/trades_style/fig_intraday_lob_volume.pdf ]]; then
    cmd="python scripts/86_intraday_lob_volume.py \
          --real-dir data/processed/val \
          --synth-base $SYNTH_BASE --models v2 v5 v9 \
          --out results/trades_style"
    jid=$(submit_one new_intraday_volume 64G 1 "" "$cmd") && {
        JIDS+=("$jid"); NAMES[$jid]="new_intraday_volume"
    }
else
    log "  SKIP intraday volume (output exists)"
fi

# ─── NEW (TRADES-style) — full return distribution (script 87) ────────────
log ""
log "── NEW — full return distribution (script 87) ──"
if [[ ! -f results/trades_style/fig_return_distribution_full.pdf ]]; then
    cmd="python scripts/87_return_distribution_full.py \
          --real-dir data/processed/val \
          --synth-base $SYNTH_BASE --models v2 v5 v9 \
          --out results/trades_style"
    jid=$(submit_one new_return_dist 64G 2 "" "$cmd") && {
        JIDS+=("$jid"); NAMES[$jid]="new_return_dist"
    }
else
    log "  SKIP return distribution (output exists)"
fi

# ─── P2.G4 — predictive-score LSTM-size sweep (3 sizes × all models) ──────
log ""
log "── P2.G4 — predictive-score LSTM-size sweep ──"
SYNTH_LIST=()
LABEL_LIST=()
for lab in "${ALL_MODELS[@]}"; do
    d="${SYNTH_DIR[$lab]}"
    if [[ -d "$d" ]] && find "$d" -name '*.parquet' -print -quit 2>/dev/null | grep -q .; then
        SYNTH_LIST+=("$d"); LABEL_LIST+=("$lab")
    fi
done
for h in 32 64 128; do
    out="results/predictive_lstm_h${h}"
    if [[ -f "$out/predictive_score.csv" ]]; then
        log "  SKIP h=$h (output exists)"
        continue
    fi
    mkdir -p "$out"
    cmd="python scripts/63_predictive_score.py \
          --real-train-dir data/processed/train \
          --real-test-dir data/processed/val \
          --synth-dirs ${SYNTH_LIST[*]} --labels ${LABEL_LIST[*]} \
          --out $out --window-size 64 --stride 8 \
          --max-tapes-train 200 --max-tapes-test 10 --seed 42 \
          --device cpu --hidden-dim $h"
    jid=$(submit_one p2g4_lstm_h${h} 64G 2 "" "$cmd") && {
        JIDS+=("$jid"); NAMES[$jid]="p2g4_lstm_h${h}"
    }
done

# ══════════════════════════════════════════════════════════════════════════
# Wait once for ALL submitted jobs
# ══════════════════════════════════════════════════════════════════════════
log ""
log "════════════════════════════════════════════════════════════════"
log "Submitted ${#JIDS[@]} jobs. Waiting for all to drain..."
log "════════════════════════════════════════════════════════════════"

if [[ ${#JIDS[@]} -eq 0 ]]; then
    log "Nothing to do — all outputs already exist."
    exit 0
fi

# Print the JID → name table once at the start
log ""
log "Job table:"
for jid in "${JIDS[@]}"; do
    log "  JID=$jid  name=${NAMES[$jid]}"
done
log ""

# Single poll loop watches all submitted jobs at once
prev_running=-1
while true; do
    running=0
    for jid in "${JIDS[@]}"; do
        if squeue -u "$USER" -j "$jid" -h 2>/dev/null | grep -q .; then
            running=$((running + 1))
        fi
    done
    if [[ $running -eq 0 ]]; then
        log "All jobs drained."
        break
    fi
    if [[ $running -ne $prev_running ]]; then
        log "  $running / ${#JIDS[@]} still running"
        prev_running=$running
    fi
    sleep 30
done

# ══════════════════════════════════════════════════════════════════════════
# Final summary — state per job
# ══════════════════════════════════════════════════════════════════════════
log ""
log "════════════════════════════════════════════════════════════════"
log "FINAL STATES"
log "════════════════════════════════════════════════════════════════"
ok=0; fail=0
for jid in "${JIDS[@]}"; do
    state=$(sacct -j "$jid" --format=State -n -P 2>/dev/null \
              | grep -v -E "^(EXTERN|BATCH)$" | head -1 | tr -d ' ')
    name="${NAMES[$jid]}"
    case "$state" in
        COMPLETED)
            log "  ✓ $name  (JID=$jid)  $state"
            ok=$((ok + 1))
            ;;
        *)
            log "  ✗ $name  (JID=$jid)  $state"
            log "    -- tail logs/${name}_${jid}.err --"
            tail -5 "logs/${name}_${jid}.err" 2>/dev/null | sed 's/^/      /'
            fail=$((fail + 1))
            ;;
    esac
done

log ""
log "$ok / ${#JIDS[@]} jobs succeeded; $fail failed"
log ""
log "Pull results to Mac:"
log "  rsync -avz favara@orcd-login.mit.edu:diff_mm/results/ ~/Desktop/diff-market-making/results/"
log ""
log "Then locally: python scripts/80_report_figures.py"
