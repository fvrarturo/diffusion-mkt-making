#!/bin/bash
# v9 / v9_b end-to-end post-training pipeline.
#
# Run in tmux AFTER kicking off the v9 + v9_b training jobs:
#     tmux new -s v9_post
#     ./scripts/utils/run_v9_post_training.sh 13515821 13515822 \
#         2>&1 | tee logs/v9_post_pipeline_$(date +%Y%m%d_%H%M%S).log
#     # Ctrl-b d to detach
#
# Sequence per model:
#   1. Wait for training to finish (or hit time limit — checkpoints saved either way)
#   2. Find newest valid checkpoint
#   3. CPU sampling (full 400-task manifest, max concurrency)
#   4. Phase 4 replay-synth → ranking_pi_diff
#   5. Phase 6 validation (CPU parts B/C/D + GPU parts E/F)
#   6. Archive results to results/_archive/<model>/
#   7. Phase 5 hypothesis test (opens holdout once for v9; reuses for v9_b)
#
# Robust to time-limit cancellation: jobs that hit the wall save partial
# checkpoints; pipeline picks the best epoch by valloss filename.

set -u  # fail on undefined; explicit error handling per stage

# ─── Configuration ──────────────────────────────────────────────────────────
V9_TRAIN_JID="${1:?v9 training job ID required as arg 1}"
V9B_TRAIN_JID="${2:?v9_b training job ID required as arg 2}"

CKPT_ROOT="$HOME/orcd/scratch/diffmm/checkpoints"
SYNTH_ROOT="$HOME/orcd/scratch/diffmm/synthetic"
COPULA_PATH="data/processed/norm_stats_INTC.copula.json"

# CPU sampling concurrency — verified at 50% chunk size during v8 sampling.
# Adjust if your QOS is different.
SAMPLE_ARRAY="--array=0-399"

# Manifest size match
MANIFEST_SEEDS=100
MANIFEST_PATH="data/manifests/ph2_sample.json"

# ─── Helpers ────────────────────────────────────────────────────────────────
log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_jobs() {
    # Poll squeue for given JIDs to all finish (any state — DONE, FAILED, CANCELLED, TIMEOUT)
    local jids="$*"
    log "Waiting for jobs: $jids"
    while true; do
        local running=0
        for jid in $jids; do
            if squeue -u favara -j "$jid" -h 2>/dev/null | grep -q .; then
                running=$((running + 1))
            fi
        done
        if [[ $running -eq 0 ]]; then
            log "All waited jobs finished"
            return 0
        fi
        sleep 60
    done
}

best_ckpt() {
    # Find best (lowest valloss) checkpoint in a run dir
    local run_dir="$1"
    ls -1 "$run_dir"/ckpt-epoch*-valloss*.ckpt 2>/dev/null \
        | awk -F'valloss' '{print $2, $0}' \
        | sort -n \
        | head -1 \
        | awk '{print $2}'
}

ensure_manifest() {
    if [[ ! -f "$MANIFEST_PATH" ]]; then
        log "Regenerating manifest with $MANIFEST_SEEDS seeds × 4 regimes"
        python -c "
import json
regimes = ['base', 'high_vol', 'toxic', 'thin']
m = [{'regime': r, 'seed': s} for r in regimes for s in range($MANIFEST_SEEDS)]
json.dump(m, open('$MANIFEST_PATH', 'w'))
print(f'wrote {len(m)} tasks')
"
    fi
    local n=$(python -c "import json; print(len(json.load(open('$MANIFEST_PATH'))))")
    log "Manifest has $n tasks"
}

# Submit a job and capture its JID; returns 1 if submission failed.
submit() {
    local jid
    jid=$(sbatch --parsable "$@" 2>&1) || { log "FAILED to submit: $*"; log "$jid"; return 1; }
    echo "$jid"
}

# ─── Pipeline stages ─────────────────────────────────────────────────────────

stage_sample() {
    local model=$1 ckpt=$2 extra_overrides=$3
    local synth_out="$SYNTH_ROOT/intc_$model"
    log "[$model] Submitting CPU sampling (output: $synth_out)"
    local jid
    jid=$(submit --exclude=node4104 $SAMPLE_ARRAY \
        slurms/ph2_sample_cpu.slurm \
        generator=ddim_sample \
        data=intc_top \
        generator.checkpoint_path="$ckpt" \
        cluster.synthetic_root="$synth_out" \
        $extra_overrides) || return 1
    log "[$model] Sampling submitted as JID=$jid"
    wait_for_jobs "$jid"
    # Verify sampling produced parquets
    local n_parquets=$(find "$synth_out" -name '*.parquet' 2>/dev/null | wc -l)
    log "[$model] Sampling done — $n_parquets parquets written"
    if [[ $n_parquets -lt 1000 ]]; then
        log "[$model] WARNING: < 1000 parquets — sampling may have largely failed"
    fi
}

stage_ph4_ph6() {
    local model=$1 ckpt=$2 extra_overrides=$3
    local synth_dir="$SYNTH_ROOT/intc_$model/INTC"
    local conditioning_type=$(echo "$extra_overrides" | grep -oE 'conditioning_type=[a-z_]*' | cut -d= -f2)
    local prediction_type=$(echo "$extra_overrides" | grep -oE 'prediction_type=[a-z_]*' | cut -d= -f2)
    [[ -z "$conditioning_type" ]] && conditioning_type=film
    [[ -z "$prediction_type" ]] && prediction_type=eps

    log "[$model] Submitting ph4 (replay-synth), ph6 (cpu+gpu) in parallel"
    local jid_ph4
    jid_ph4=$(submit slurms/ph4_replay_synth.slurm \
        data=intc_top \
        +synthetic_root="$SYNTH_ROOT/intc_$model") || return 1

    local jid_ph6cpu
    jid_ph6cpu=$(SYNTH_DIR="$synth_dir" N_SYNTH=250 \
        submit slurms/ph6_validate_cpu.slurm) || return 1

    local jid_ph6gpu
    jid_ph6gpu=$(CKPT="$ckpt" SYNTH_DIR="$synth_dir" \
        USE_COPULA=1 PREDICTION_TYPE="$prediction_type" \
        CONDITIONING_TYPE="$conditioning_type" USE_EDM=0 \
        submit slurms/ph6_validate_gpu.slurm) || return 1

    log "[$model] ph4=$jid_ph4 ph6_cpu=$jid_ph6cpu ph6_gpu=$jid_ph6gpu"
    wait_for_jobs "$jid_ph4 $jid_ph6cpu $jid_ph6gpu"
    log "[$model] ph4 + ph6 done"
}

stage_archive() {
    local model=$1
    local arch="results/_archive/$model"
    mkdir -p "$arch"
    log "[$model] Archiving results to $arch/"
    [[ -d results/backtests/synthetic/INTC ]] && cp -r results/backtests/synthetic/INTC "$arch/synthetic_backtests"
    [[ -d results/validation/INTC ]] && cp -r results/validation/INTC "$arch/validation"
    log "[$model] Archive: $(find "$arch" -type f | wc -l) files"
}

stage_hypothesis() {
    local model=$1
    local arch="results/_archive/$model"
    # Restore this model's synth backtests so ph5 reads the right ranking_pi_diff
    log "[$model] Restoring synth backtests for ph5 hypothesis test"
    if [[ -d "$arch/synthetic_backtests" ]]; then
        rm -rf results/backtests/synthetic/INTC
        cp -r "$arch/synthetic_backtests" results/backtests/synthetic/INTC
    fi
    log "[$model] Submitting ph5 hypothesis test"
    local jid
    jid=$(submit slurms/ph5_hypothesis.slurm data=intc_top) || return 1
    log "[$model] ph5 submitted as JID=$jid"
    wait_for_jobs "$jid"
    # Archive the hypothesis output
    [[ -d results/hypothesis/INTC ]] && cp -r results/hypothesis/INTC "$arch/hypothesis"
    [[ -d results/backtests/holdout/INTC ]] && cp -r results/backtests/holdout/INTC "$arch/holdout"
    log "[$model] Hypothesis test done; results in $arch/hypothesis/"
    if [[ -f "$arch/hypothesis/bootstrap.json" ]]; then
        echo "  ─── $model bootstrap.json ───"
        python -m json.tool "$arch/hypothesis/bootstrap.json" | head -20
    fi
}

# ─── Main ───────────────────────────────────────────────────────────────────

log "=== v9 post-training orchestrator ==="
log "v9 train JID:   $V9_TRAIN_JID"
log "v9_b train JID: $V9B_TRAIN_JID"

# 1. Wait for training to finish
wait_for_jobs "$V9_TRAIN_JID $V9B_TRAIN_JID"

# 2. Locate best checkpoints
V9_CKPT=$(best_ckpt "$CKPT_ROOT/intc_v9")
V9B_CKPT=$(best_ckpt "$CKPT_ROOT/intc_v9_b")
log "v9 best ckpt:   $V9_CKPT"
log "v9_b best ckpt: $V9B_CKPT"
if [[ -z "$V9_CKPT" || -z "$V9B_CKPT" ]]; then
    log "FATAL: missing checkpoint; aborting"
    exit 1
fi

# 3. Verify copula exists (should be, since training fitted it)
if [[ ! -f "$COPULA_PATH" ]]; then
    log "FATAL: copula transform $COPULA_PATH not found"
    exit 1
fi
log "Copula JSON OK: $COPULA_PATH"

# 4. Ensure manifest is the right size
ensure_manifest

# 5. Sample v9 first (sequential to avoid QOS contention from two array jobs)
log "─── Stage: sample v9 ───"
stage_sample "v9" "$V9_CKPT" ""

log "─── Stage: ph4 + ph6 v9 ───"
stage_ph4_ph6 "v9" "$V9_CKPT" "PREDICTION_TYPE=eps CONDITIONING_TYPE=film"

log "─── Stage: archive v9 ───"
stage_archive "v9"

# 6. Same for v9_b
log "─── Stage: sample v9_b ───"
stage_sample "v9_b" "$V9B_CKPT" "generator.schedule.prediction_type=v"

log "─── Stage: ph4 + ph6 v9_b ───"
stage_ph4_ph6 "v9_b" "$V9B_CKPT" "PREDICTION_TYPE=v CONDITIONING_TYPE=film"

log "─── Stage: archive v9_b ───"
stage_archive "v9_b"

# 7. Hypothesis tests (sequential, with archive restore between)
log "─── Stage: ph5 hypothesis v9 ───"
stage_hypothesis "v9"

log "─── Stage: ph5 hypothesis v9_b ───"
stage_hypothesis "v9_b"

# 8. Final side-by-side diagnostics
log "─── Stage: deep diagnostic across v2 / v8 / v9 / v9_b ───"
python scripts/60_deep_diagnostic.py \
    --real-dir data/processed/val \
    --synth-dirs \
        $HOME/orcd/scratch/diffmm/synthetic_v2_stitched/INTC \
        $HOME/orcd/scratch/diffmm/synthetic/intc_v8/INTC \
        $HOME/orcd/scratch/diffmm/synthetic/intc_v9/INTC \
        $HOME/orcd/scratch/diffmm/synthetic/intc_v9_b/INTC \
    --labels v2 v8 v9 v9_b \
    --copula-json "$COPULA_PATH" \
    --copula-models v8 v9 v9_b \
    --out results/deep_diagnostic_v9 \
    --max-tapes 100 || log "WARN: deep diagnostic failed (non-fatal)"

log "─── Stage: ranking-mechanism diagnostic across all models ───"
python scripts/61_ranking_mechanism.py \
    --metrics \
        truth:results/_archive/v9/holdout/metrics.csv \
        v8:results/_archive/v8/synthetic_backtests/metrics.csv \
        v8_b:results/_archive/v8_b/synthetic_backtests/metrics.csv \
        v9:results/_archive/v9/synthetic_backtests/metrics.csv \
        v9_b:results/_archive/v9_b/synthetic_backtests/metrics.csv \
    --real-dir data/processed/val \
    --synth-dirs \
        v2:$HOME/orcd/scratch/diffmm/synthetic_v2_stitched/INTC \
        v8:$HOME/orcd/scratch/diffmm/synthetic/intc_v8/INTC \
        v9:$HOME/orcd/scratch/diffmm/synthetic/intc_v9/INTC \
        v9_b:$HOME/orcd/scratch/diffmm/synthetic/intc_v9_b/INTC \
    --out results/ranking_mechanism_v9 \
    --max-files 50 || log "WARN: ranking-mechanism diagnostic failed (non-fatal)"

log "=== ALL DONE ==="
log "Headline numbers:"
for m in v9 v9_b; do
    [[ -f "results/_archive/$m/hypothesis/bootstrap.json" ]] && {
        echo "  $m:"
        python -c "
import json
d = json.load(open('results/_archive/$m/hypothesis/bootstrap.json'))
print(f'    rho_diff={d.get(\"rho_diff_point\", \"?\")}  rho_hist={d.get(\"rho_hist_point\", \"?\")}  p={d.get(\"p_value_one_sided\", \"?\")}')
print(f'    pi_diff={d.get(\"pi_diff\", \"?\")}')"
    }
done
