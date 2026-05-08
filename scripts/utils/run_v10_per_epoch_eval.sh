#!/bin/bash
# Phase E Experiment 2 — per-epoch evaluation of v10 (v2 → v-prediction fine-tune).
#
# Maps the marginals-vs-joints tradeoff curve. After v10 finishes training (with
# every_n_epochs=1, save_top_k=-1), this script samples + ph4 + ph5 at each saved
# epoch checkpoint, archiving results so we can plot (kurtosis, ρ_diff) per epoch.
#
# Usage (in tmux, after v10 training finishes):
#   tmux new -s v10_eval
#   ./scripts/utils/run_v10_per_epoch_eval.sh \
#       2>&1 | tee logs/v10_per_epoch_$(date +%Y%m%d_%H%M%S).log
#
# Per epoch:
#   1. Sample on CPU partition (full 400-task manifest) → unique synth dir
#   2. Run ph4_replay_synth → ranking_pi_diff for that epoch
#   3. Archive backtest metrics + ranking
#   4. Run ph5_hypothesis (uses already-opened holdout from v9 run)
#   5. Archive hypothesis output
#
# Skips ph6 (validation suite) by default — we only need the headline ρ_diff
# trajectory for the figure. If you want full validation, uncomment the ph6 stage.

set -u

CKPT_DIR="${CKPT_DIR:-$HOME/orcd/scratch/diffmm/checkpoints/intc_v10}"
SYNTH_ROOT="${SYNTH_ROOT:-$HOME/orcd/scratch/diffmm/synthetic}"
EPOCHS="${EPOCHS:-1 2 3 5}"        # which saved epochs to evaluate
SAMPLE_ARRAY="${SAMPLE_ARRAY:---array=0-399}"
RUN_PH6="${RUN_PH6:-0}"             # set to 1 to also run ph6 validation

log() { echo "[$(date '+%H:%M:%S')] $*"; }

wait_for_jobs() {
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
            log "  all waited jobs finished"; return 0
        fi
        sleep 60
    done
}

submit() {
    local jid
    jid=$(sbatch --parsable "$@" 2>&1) || { log "FAILED to submit: $*"; log "$jid"; return 1; }
    echo "$jid"
}

ckpt_for_epoch() {
    # Find the checkpoint file for a specific epoch. Lightning naming:
    # ckpt-epoch003-valloss0.0123.ckpt
    local epoch=$1
    local epoch_str=$(printf "%03d" $epoch)
    ls -1 "$CKPT_DIR"/ckpt-epoch${epoch_str}-valloss*.ckpt 2>/dev/null | head -1
}

# ─── Main ───────────────────────────────────────────────────────────────────

log "=== v10 per-epoch evaluation orchestrator ==="
log "checkpoint dir: $CKPT_DIR"
log "epochs to evaluate: $EPOCHS"
log "ph6 validation: $([[ $RUN_PH6 == 1 ]] && echo enabled || echo SKIPPED)"

if [[ ! -d "$CKPT_DIR" ]]; then
    log "FATAL: checkpoint dir $CKPT_DIR not found"
    exit 1
fi

ls -1 "$CKPT_DIR"/ckpt-*.ckpt 2>/dev/null | head -10
echo

# Pre-flight: confirm all requested epochs have checkpoints
for epoch in $EPOCHS; do
    ckpt=$(ckpt_for_epoch "$epoch")
    if [[ -z "$ckpt" ]]; then
        log "FATAL: no checkpoint for epoch $epoch in $CKPT_DIR"
        exit 1
    fi
    log "  epoch $epoch: $ckpt"
done

# Per-epoch loop
for epoch in $EPOCHS; do
    ckpt=$(ckpt_for_epoch "$epoch")
    epoch_str=$(printf "%03d" $epoch)
    model_label="v10_e${epoch_str}"
    synth_out="$SYNTH_ROOT/intc_${model_label}"
    arch="results/_archive/${model_label}"

    log "════════════════════════════════════════════════════════════════"
    log "═══ EPOCH $epoch — $model_label"
    log "════════════════════════════════════════════════════════════════"

    # 1. Sample
    log "[$model_label] Submitting CPU sampling → $synth_out"
    sample_jid=$(submit --exclude=node4104 $SAMPLE_ARRAY \
        slurms/ph2_sample_cpu.slurm \
        generator=ddim_sample \
        data=intc_top \
        generator.checkpoint_path="$ckpt" \
        generator.schedule.prediction_type=v \
        cluster.synthetic_root="$synth_out") || { log "skip $model_label"; continue; }
    log "[$model_label] sampling JID=$sample_jid"
    wait_for_jobs "$sample_jid"

    n_parquets=$(find "$synth_out" -name '*.parquet' 2>/dev/null | wc -l)
    log "[$model_label] sampling done — $n_parquets parquets"

    # 2. ph4 replay
    log "[$model_label] Submitting ph4 replay-synth"
    ph4_jid=$(submit slurms/ph4_replay_synth.slurm \
        data=intc_top \
        +synthetic_root="$synth_out") || { log "skip $model_label ph4"; continue; }

    # 2b. (optional) ph6 validation
    if [[ $RUN_PH6 == 1 ]]; then
        log "[$model_label] Submitting ph6 validation (cpu+gpu)"
        ph6cpu_jid=$(SYNTH_DIR="$synth_out/INTC" N_SYNTH=250 \
            submit slurms/ph6_validate_cpu.slurm) || ph6cpu_jid=""
        ph6gpu_jid=$(CKPT="$ckpt" SYNTH_DIR="$synth_out/INTC" \
            USE_COPULA=0 PREDICTION_TYPE=v CONDITIONING_TYPE=film USE_EDM=0 \
            submit slurms/ph6_validate_gpu.slurm) || ph6gpu_jid=""
        wait_for_jobs "$ph4_jid $ph6cpu_jid $ph6gpu_jid"
    else
        wait_for_jobs "$ph4_jid"
    fi

    # 3. Archive
    mkdir -p "$arch"
    [[ -d results/backtests/synthetic/INTC ]] && cp -r results/backtests/synthetic/INTC "$arch/synthetic_backtests"
    [[ $RUN_PH6 == 1 && -d results/validation/INTC ]] && cp -r results/validation/INTC "$arch/validation"
    log "[$model_label] archived to $arch ($(find $arch -type f | wc -l) files)"

    # 4. ph5 hypothesis (reuses opened holdout)
    log "[$model_label] Submitting ph5 hypothesis"
    ph5_jid=$(submit slurms/ph5_hypothesis.slurm data=intc_top) || { log "skip $model_label ph5"; continue; }
    wait_for_jobs "$ph5_jid"
    [[ -d results/hypothesis/INTC ]] && cp -r results/hypothesis/INTC "$arch/hypothesis"
    log "[$model_label] ph5 done"

    if [[ -f "$arch/hypothesis/bootstrap.json" ]]; then
        echo "  ─── $model_label bootstrap.json ───"
        python -m json.tool "$arch/hypothesis/bootstrap.json" | head -15
    fi
done

log "════════════════════════════════════════════════════════════════"
log "=== ALL EPOCHS DONE ==="
log "════════════════════════════════════════════════════════════════"

# Summary table
echo ""
echo "Per-epoch trajectory (kurtosis, ρ_diff):"
printf "%-12s %-10s %-10s %-30s\n" "model" "rho_diff" "rho_hist" "pi_diff"
for epoch in 0 $EPOCHS; do
    if [[ $epoch == 0 ]]; then
        # v2 baseline
        bs="results/_archive/v2/hypothesis/bootstrap.json"
        label="v2 (e=0)"
    else
        epoch_str=$(printf "%03d" $epoch)
        bs="results/_archive/v10_e${epoch_str}/hypothesis/bootstrap.json"
        label="v10 (e=$epoch)"
    fi
    if [[ -f "$bs" ]]; then
        rho_diff=$(python -c "import json; d=json.load(open('$bs')); print(d.get('rho_diff_point','?'))")
        rho_hist=$(python -c "import json; d=json.load(open('$bs')); print(d.get('rho_hist_point','?'))")
        pi_diff=$(python -c "import json; d=json.load(open('$bs')); print(d.get('pi_diff','?'))")
        printf "%-12s %-10s %-10s %-30s\n" "$label" "$rho_diff" "$rho_hist" "$pi_diff"
    else
        printf "%-12s %-10s %-10s %-30s\n" "$label" "-" "-" "(no bootstrap.json)"
    fi
done
