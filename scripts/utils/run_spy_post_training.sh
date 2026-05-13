#!/bin/bash
# SPY post-training pipeline — sampling + validation + cross-ticker eval,
# all fanned out to mit_preemptable. Picks up after the cross-ticker
# orchestrator's training stage completes (and works around its
# wait-too-early bug by polling the actual JIDs / checkpoints itself).
#
# Run on cluster login node:
#   tmux new -s spy_post
#   ./scripts/utils/run_spy_post_training.sh \
#       2>&1 | tee logs/spy_post_$(date +%Y%m%d_%H%M%S).log
#   # Ctrl-b d
#
# Stages:
#   1. Wait for spy_v2 + spy_v5 trainings to finish (or detect they have)
#   2. Find best checkpoints
#   3. Sample (400-task arrays per model) on mit_preemptable — both in parallel
#   4. Validate B/C/D (script 50) on mit_preemptable — both in parallel
#   5. TRADES metrics (script 62) + predictive score (script 63) on
#      mit_preemptable — both in parallel
#   6. Final summary
#
# Wall time on preemptable: ~30-45 min total (sampling is the long pole).

set -uo pipefail

CKPT_ROOT="$HOME/orcd/scratch/diffmm/checkpoints"
SYNTH_ROOT="$HOME/orcd/scratch/diffmm/synthetic"
TICKER="SPY"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p logs

best_ckpt() {
    # Lowest-valloss checkpoint anywhere under the run dir. Lightning typically
    # writes to <run_dir>/default/ckpt-epoch*-valloss*.ckpt; we also look in the
    # run dir itself for backwards compatibility with the older v9-era runs.
    #
    # FALLBACK for chained-training (SPY): if the chunked 6h jobs never
    # completed an epoch (each chunk hit its time budget on step-based
    # saves), no ckpt-epoch*-valloss*.ckpt files exist. In that case we
    # fall back to last.ckpt (Lightning maintains it as the truly latest
    # state) or to the highest-numbered ckpt-step*.ckpt.
    local d="$1"
    [[ -d "$d" ]] || return 0
    local c
    c=$( {
        ls -1 "$d"/ckpt-epoch*-valloss*.ckpt 2>/dev/null
        ls -1 "$d"/default/ckpt-epoch*-valloss*.ckpt 2>/dev/null
        ls -1 "$d"/lightning_logs/version_*/checkpoints/ckpt-epoch*-valloss*.ckpt 2>/dev/null
    } | awk -F'valloss' '{print $2, $0}' \
      | sort -n | head -1 | awk '{print $2}' )
    if [[ -n "$c" ]]; then
        echo "$c"
        return 0
    fi
    # Fallback path
    {
        ls -1 "$d"/last.ckpt 2>/dev/null
        ls -1 "$d"/default/last.ckpt 2>/dev/null
        ls -1 "$d"/lightning_logs/version_*/checkpoints/last.ckpt 2>/dev/null
        ls -1 "$d"/ckpt-step*.ckpt 2>/dev/null            | sort -V | tail -1
        ls -1 "$d"/default/ckpt-step*.ckpt 2>/dev/null    | sort -V | tail -1
        ls -1 "$d"/lightning_logs/version_*/checkpoints/ckpt-step*.ckpt 2>/dev/null | sort -V | tail -1
    } | head -1
}

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

wait_jids() {
    local jids="$*"
    [[ -z "$jids" ]] && return 0
    local prev=-1
    while true; do
        local running=0 total=0
        for jid in $jids; do
            total=$((total + 1))
            if squeue -u "$USER" -j "$jid" -h 2>/dev/null | grep -q .; then
                running=$((running + 1))
            fi
        done
        if [[ $running -eq 0 ]]; then
            log "  all $total jobs drained"
            return 0
        fi
        if [[ $running -ne $prev ]]; then
            log "  $running / $total still running"
            prev=$running
        fi
        sleep 30
    done
}

# ══════════════════════════════════════════════════════════════════════════
# STAGE 1 — Wait for SPY trainings (or skip if already done)
# ══════════════════════════════════════════════════════════════════════════
log "════════════════════════════════════════════════════════════════"
log "STAGE 1 — wait for spy_v2 + spy_v5 trainings"
log "════════════════════════════════════════════════════════════════"

# Find any RUNNING/PENDING ph2_finetune jobs on spy_* checkpoints
TRAIN_JIDS=$(squeue -u "$USER" -h -o "%i %j" \
    | awk '/diffmm_ph2_finetune/ {print $1}' | tr '\n' ' ')
if [[ -n "$TRAIN_JIDS" ]]; then
    log "Waiting for active ph2_finetune jobs: $TRAIN_JIDS"
    wait_jids $TRAIN_JIDS
else
    log "No active training jobs — assuming trainings already finished"
fi

# ══════════════════════════════════════════════════════════════════════════
# STAGE 2 — Find best checkpoints
# ══════════════════════════════════════════════════════════════════════════
log ""
log "════════════════════════════════════════════════════════════════"
log "STAGE 2 — locate best checkpoints"
log "════════════════════════════════════════════════════════════════"

declare -A CKPT
for model in v2 v5; do
    c=$(best_ckpt "$CKPT_ROOT/spy_$model")
    if [[ -z "$c" ]]; then
        log "  ✗ spy_$model: NO CHECKPOINT under $CKPT_ROOT/spy_$model"
        continue
    fi
    CKPT[$model]="$c"
    log "  ✓ spy_$model: $c"
done

n_ckpt=${#CKPT[@]}
if [[ $n_ckpt -eq 0 ]]; then
    log "FATAL: no checkpoints found — aborting"
    log "  Diagnostic: run 'find ~/orcd/scratch/diffmm/checkpoints -name \"*.ckpt\" -mtime -1' to locate them"
    exit 1
fi

# ══════════════════════════════════════════════════════════════════════════
# STAGE 3 — Sample (400-task array per model on mit_preemptable)
# ══════════════════════════════════════════════════════════════════════════
log ""
log "════════════════════════════════════════════════════════════════"
log "STAGE 3 — sampling (400 tasks per model on mit_preemptable)"
log "════════════════════════════════════════════════════════════════"

# Architecture flags per model
declare -A PRED_TYPE=( [v2]=eps [v5]=v )
declare -A COND_TYPE=( [v2]=film [v5]=adaln_zero )

# Ensure manifest exists
MANIFEST=data/manifests/ph2_sample.json
if [[ ! -f "$MANIFEST" ]]; then
    log "Generating sampling manifest"
    python scripts/utils/make_manifest.py --kind sample --n-seeds 100 --out "$MANIFEST"
fi

SAMPLE_JIDS=""
for model in "${!CKPT[@]}"; do
    synth_root="$SYNTH_ROOT/spy_$model"
    synth_dir="$synth_root/$TICKER"
    n_existing=$(find "$synth_dir" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')
    if [[ $n_existing -ge 1000 ]]; then
        log "  [spy_$model] $n_existing parquets already exist — skipping sample"
        continue
    fi
    log "  [spy_$model] submitting 400-task sampling array"
    jid=$(sbatch --parsable \
        --partition=mit_preemptable \
        --array=0-399 \
        --job-name=spy_${model}_sample \
        --output=logs/spy_${model}_sample_%A_%a.out \
        --error=logs/spy_${model}_sample_%A_%a.err \
        slurms/ph2_sample_cpu.slurm \
        generator=ddim_sample \
        data=spy_top \
        generator.checkpoint_path="${CKPT[$model]}" \
        generator.schedule.prediction_type="${PRED_TYPE[$model]}" \
        generator.model.conditioning_type="${COND_TYPE[$model]}" \
        cluster.synthetic_root="$synth_root" 2>/dev/null)
    if [[ -n "$jid" ]]; then
        log "    JID=$jid"
        SAMPLE_JIDS="$SAMPLE_JIDS $jid"
    else
        log "    ERROR: sample submit failed"
    fi
done

if [[ -n "$SAMPLE_JIDS" ]]; then
    log ""
    log "Waiting for sampling arrays to drain..."
    wait_jids $SAMPLE_JIDS
fi

# Verify sampling outputs
for model in "${!CKPT[@]}"; do
    synth_dir="$SYNTH_ROOT/spy_$model/$TICKER"
    n=$(find "$synth_dir" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')
    log "  [spy_$model] $n parquets in $synth_dir"
    if [[ $n -lt 100 ]]; then
        log "    WARN: few parquets — sampling may have largely failed"
    fi
done

# ══════════════════════════════════════════════════════════════════════════
# STAGE 4 — Validation (B/C/D) on mit_preemptable, both in parallel
# ══════════════════════════════════════════════════════════════════════════
log ""
log "════════════════════════════════════════════════════════════════"
log "STAGE 4 — validation (B/C/D) for both models in parallel"
log "════════════════════════════════════════════════════════════════"

VAL_JIDS=""
declare -A VAL_NAMES
for model in "${!CKPT[@]}"; do
    synth_dir="$SYNTH_ROOT/spy_$model/$TICKER"
    out_dir="results/_archive/spy_$model/validation"
    n=$(find "$synth_dir" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')
    if [[ $n -lt 100 ]]; then
        log "  [spy_$model] insufficient synth — skipping validation"
        continue
    fi
    if [[ -f "$out_dir/G1_pass_fail_summary.csv" ]]; then
        log "  [spy_$model] validation output exists — skipping"
        continue
    fi
    mkdir -p "$out_dir"
    cmd="SYNTH_DIR=$synth_dir OUT_DIR=$out_dir TICKER=$TICKER N_SYNTH=250 \
         sbatch --partition=mit_preemptable --mem=64G --time=01:00:00 \
                --cpus-per-task=4 \
                --job-name=spy_${model}_validate \
                --output=logs/spy_${model}_validate_%j.out \
                --error=logs/spy_${model}_validate_%j.err \
                slurms/ph6_validate_cpu.slurm"
    log "  EXEC: $cmd"
    jid=$(SYNTH_DIR="$synth_dir" OUT_DIR="$out_dir" TICKER="$TICKER" N_SYNTH=250 \
        sbatch --parsable \
            --partition=mit_preemptable --mem=64G --time=01:00:00 \
            --cpus-per-task=4 \
            --job-name=spy_${model}_validate \
            --output=logs/spy_${model}_validate_%j.out \
            --error=logs/spy_${model}_validate_%j.err \
            slurms/ph6_validate_cpu.slurm 2>/dev/null)
    if [[ -n "$jid" ]]; then
        log "    JID=$jid"
        VAL_JIDS="$VAL_JIDS $jid"
        VAL_NAMES[$jid]="spy_${model}_validate"
    fi
done

if [[ -n "$VAL_JIDS" ]]; then
    log ""
    log "Waiting for validation jobs to drain..."
    wait_jids $VAL_JIDS
fi

# ══════════════════════════════════════════════════════════════════════════
# STAGE 5 — Cross-model TRADES + predictive (parallel on preemptable)
# ══════════════════════════════════════════════════════════════════════════
log ""
log "════════════════════════════════════════════════════════════════"
log "STAGE 5 — TRADES metrics + predictive score (parallel)"
log "════════════════════════════════════════════════════════════════"

# Build per-ticker symlink dirs for real data (scripts 62/63 don't filter by ticker)
PROCESSED_DIR="data/processed"
for split in train val; do
    dst="$PROCESSED_DIR/${split}_${TICKER}"
    mkdir -p "$dst"
    find "$dst" -maxdepth 1 -type l -delete 2>/dev/null || true
    n=0
    for f in "$PROCESSED_DIR/$split"/${TICKER}_*.parquet; do
        [[ -e "$f" ]] || continue
        ln -sf "$(realpath "$f")" "$dst/$(basename "$f")"
        n=$((n + 1))
    done
    log "  staged $n $split parquets → $dst"
done

# Collect (label, dir) pairs whose synth dirs are populated
LABELS=()
DIRS=()
for model in "${!CKPT[@]}"; do
    d="$SYNTH_ROOT/spy_$model/$TICKER"
    n=$(find "$d" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')
    if [[ $n -ge 100 ]]; then
        LABELS+=("$model")
        DIRS+=("$d")
    fi
done

POST_JIDS=""
if [[ ${#LABELS[@]} -gt 0 ]]; then
    out_trades="results/trades_metrics_spy"
    out_pred="results/predictive_score_spy"
    mkdir -p "$out_trades" "$out_pred"

    cmd_t="python scripts/62_trades_metrics.py \
            --real-dir $PROCESSED_DIR/val_$TICKER \
            --synth-dirs ${DIRS[*]} --labels ${LABELS[*]} \
            --out $out_trades --max-tapes 60"
    jid=$(submit_one spy_trades_metrics 128G 2 "" "$cmd_t") && POST_JIDS="$POST_JIDS $jid"

    cmd_p="python scripts/63_predictive_score.py \
            --real-train-dir $PROCESSED_DIR/train_$TICKER \
            --real-test-dir $PROCESSED_DIR/val_$TICKER \
            --synth-dirs ${DIRS[*]} --labels ${LABELS[*]} \
            --out $out_pred --window-size 64 --stride 8 \
            --max-tapes-train 200 --max-tapes-test 10 --seed 42 --device cpu"
    jid=$(submit_one spy_predictive_score 64G 2 "" "$cmd_p") && POST_JIDS="$POST_JIDS $jid"
fi

if [[ -n "$POST_JIDS" ]]; then
    log ""
    log "Waiting for TRADES + predictive jobs to drain..."
    wait_jids $POST_JIDS
fi

# ══════════════════════════════════════════════════════════════════════════
# STAGE 6 — Summary
# ══════════════════════════════════════════════════════════════════════════
log ""
log "════════════════════════════════════════════════════════════════"
log "STAGE 6 — final state"
log "════════════════════════════════════════════════════════════════"
for model in v2 v5; do
    val_csv="results/_archive/spy_$model/validation/G1_pass_fail_summary.csv"
    n_synth=$(find "$SYNTH_ROOT/spy_$model/$TICKER" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')
    [[ -f "$val_csv" ]] && val_status="✓ ${val_csv}" || val_status="✗ MISSING"
    log "  spy_$model:  $n_synth synth parquets   |  validation: $val_status"
done
log ""
log "Cross-ticker outputs:"
for f in results/trades_metrics_spy/{trades_metrics_summary.txt,pca_coverage.csv} \
         results/predictive_score_spy/predictive_score.csv; do
    [[ -f "$f" ]] && log "  ✓ $f" || log "  ✗ $f MISSING"
done
log ""
log "Pull to Mac:"
log "  rsync -avz favara@orcd-login.mit.edu:diff_mm/results/{_archive/spy_*,trades_metrics_spy,predictive_score_spy}/ \\"
log "      ~/Desktop/diff-market-making/results/"
