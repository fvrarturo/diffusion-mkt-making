#!/bin/bash
# SPY chained-resume training for esampi — recovers from the 6h-cap-vs-epoch
# mismatch that left the previous SPY runs with zero checkpoints.
#
# Strategy:
#   1. Submit 4 sequential 6h GPU jobs PER MODEL on mit_normal_gpu.
#      Job 2-4 each depend on the previous job (--dependency=afterany) and
#      auto-discover the latest checkpoint at start time so they resume from
#      where the previous job left off.
#   2. Lightning's `+resume_from=<ckpt>` restores optimizer + scheduler +
#      epoch + global_step (verified by run_v9_post_training.sh comments).
#   3. NEW: every job has `+generator.checkpointing.every_n_train_steps=10000`
#      so we get periodic saves even when a single job doesn't complete an
#      epoch. The patched scripts/11_finetune_generator.py handles the new
#      key (defaults to disabled when not set).
#   4. v2 chain (4 jobs) and v5 chain (4 jobs) run independently — both
#      get submitted up-front so SLURM schedules them in whatever order it
#      can, but each chain is internally serial via dependencies.
#   5. After all 8 jobs drain (or hit final TIMEOUT), pick the best
#      (lowest val/loss) checkpoint per model and run the standard SPY
#      post-training pipeline (sample → validate → predictive → TRADES).
#
# Run on cluster login node (must be one with working scratch mount, e.g. login001):
#   tmux new -s spy_chain
#   ./scripts/utils/run_spy_chained_training.sh \
#       2>&1 | tee logs/spy_chain_$(date +%Y%m%d_%H%M%S).log
#   # Ctrl-b d
#
# Total wall time: 4 × 6h = 24h compute per model, but jobs run in parallel
# across two models; if v2 + v5 schedule simultaneously, ~24h end-to-end.
# Add ~1.5h for the post-training pipeline → expect everything done in 26-30h.

set -uo pipefail

CKPT_ROOT="$HOME/orcd/scratch/diffmm/checkpoints"
SYNTH_ROOT="$HOME/orcd/scratch/diffmm/synthetic"
TICKER="SPY"

log() { echo "[$(date '+%H:%M:%S')] $*"; }
mkdir -p logs

# ─── Architecture overrides per model ─────────────────────────────────────
declare -A PRED_TYPE=( [v2]=eps [v5]=v          )
declare -A COND_TYPE=( [v2]=film [v5]=adaln_zero )

# ─── Verify scratch is mounted (we got bitten by login008 earlier) ────────
if [[ ! -d "$CKPT_ROOT" ]]; then
    log "FATAL: $CKPT_ROOT not visible from this login node"
    log "  Try ssh'ing to a different login (e.g. orcd-login001) and re-run"
    exit 1
fi
log "scratch mount OK: $CKPT_ROOT exists"

# ─── Build per-model 4-job chain ──────────────────────────────────────────
N_CHUNKS=4   # 4 × 6h = 24h compute per model

declare -A FINAL_JIDS   # last JID in each model's chain (what to wait on)

for model in v2 v5; do
    pred="${PRED_TYPE[$model]}"
    cond="${COND_TYPE[$model]}"
    ckpt_dir="$CKPT_ROOT/spy_$model"

    log ""
    log "════════════════════════════════════════════════════════════════"
    log "Building $N_CHUNKS-chunk training chain for spy_$model "
    log "  (pred=$pred  cond=$cond  ckpt_dir=$ckpt_dir)"
    log "════════════════════════════════════════════════════════════════"

    prev_jid=""
    for chunk in $(seq 1 $N_CHUNKS); do
        # Build the wrapper command. Chunks 2+ auto-discover the latest
        # checkpoint and pass it as +resume_from=. Chunk 1 starts fresh.
        # We do the discovery INSIDE the SLURM job (not at submit time)
        # because chunks 2+ run after the previous chunk finishes — so the
        # latest ckpt only exists at runtime, not at submit time.
        if [[ $chunk -eq 1 ]]; then
            resume_logic="RESUME_FLAG=\"\""
        else
            # Pick latest "last.ckpt" if it exists (Lightning's stable resume
            # target), else the most-recently-modified ckpt-step*.ckpt.
            resume_logic='
            CKPT_DIR="'"$ckpt_dir"'/default"
            LATEST=$(ls -t "$CKPT_DIR"/last.ckpt "$CKPT_DIR"/ckpt-step*.ckpt "$CKPT_DIR"/ckpt-epoch*.ckpt 2>/dev/null | head -1)
            if [[ -z "$LATEST" ]]; then
                echo "FATAL: chunk '$chunk' has no checkpoint to resume from in $CKPT_DIR" >&2
                exit 1
            fi
            echo "RESUMING from: $LATEST"
            RESUME_FLAG="+resume_from=$LATEST"'
        fi

        # Build the python command
        py_cmd="python scripts/11_finetune_generator.py \
            cluster=engaging \
            data=spy_top \
            generator.schedule.prediction_type=$pred \
            generator.model.conditioning_type=$cond \
            cluster.checkpoint_root=$ckpt_dir \
            +generator.checkpointing.every_n_train_steps=10000 \
            \$RESUME_FLAG"

        # Build sbatch dependency flag
        dep_flag=""
        if [[ -n "$prev_jid" ]]; then
            dep_flag="--dependency=afterany:$prev_jid"
        fi

        log "  chunk $chunk:"
        if [[ -n "$prev_jid" ]]; then
            log "    depends on JID $prev_jid (afterany)"
        else
            log "    no dependency (initial chunk)"
        fi
        log "    EXEC (sbatch): ph2_finetune.slurm with every_n_train_steps=10000 $([[ $chunk -gt 1 ]] && echo "+ resume from latest")"

        jid=$(sbatch --parsable \
            $dep_flag \
            --partition=mit_normal_gpu \
            --gres=gpu:1 \
            --cpus-per-task=8 \
            --mem=64G \
            --time=06:00:00 \
            --job-name="spy_${model}_c${chunk}" \
            --output="logs/spy_${model}_c${chunk}_%j.out" \
            --error="logs/spy_${model}_c${chunk}_%j.err" \
            --wrap="source slurms/_common.sh 2>/dev/null || true; cd $PWD; $resume_logic; eval $py_cmd" 2>&1)
        if [[ -z "$jid" || ! "$jid" =~ ^[0-9]+$ ]]; then
            log "    ERROR: sbatch failed: $jid"
            exit 1
        fi
        log "    JID=$jid"
        prev_jid="$jid"
    done
    FINAL_JIDS[$model]="$prev_jid"
done

log ""
log "════════════════════════════════════════════════════════════════"
log "All $((N_CHUNKS * 2)) chained-training jobs submitted."
log "  spy_v2 final JID: ${FINAL_JIDS[v2]}"
log "  spy_v5 final JID: ${FINAL_JIDS[v5]}"
log "════════════════════════════════════════════════════════════════"
log ""
log "Detach now (Ctrl-b d) and check back in ~24h."
log ""
log "Mid-run progress checks:"
log "  squeue -u \$USER                        # which chunk is running"
log "  ls -lh $CKPT_ROOT/spy_v{2,5}/default/    # checkpoints accumulating"
log "  tail logs/spy_v{2,5}_c\$N_*.out          # training progress per chunk"
log ""
log "═══ Waiting for both chains to drain ═══"

# Poll until BOTH final JIDs are done
ALL_FINAL="${FINAL_JIDS[v2]} ${FINAL_JIDS[v5]}"
prev_running=-1
while true; do
    running=0
    for jid in $ALL_FINAL; do
        # Note: this only checks the FINAL chunk JID. Earlier chunks may still
        # be in dependency-PENDING; we want to wait for the entire chain end.
        if squeue -u "$USER" -j "$jid" -h 2>/dev/null | grep -q .; then
            running=$((running + 1))
        fi
    done
    if [[ $running -eq 0 ]]; then
        log "Both chains' final chunks finished."
        break
    fi
    if [[ $running -ne $prev_running ]]; then
        log "  $running / 2 final chunks still queued/running"
        prev_running=$running
    fi
    sleep 120
done

# ══════════════════════════════════════════════════════════════════════════
# Best-checkpoint pick + standard post-training pipeline
# ══════════════════════════════════════════════════════════════════════════
log ""
log "════════════════════════════════════════════════════════════════"
log "All chains drained — picking best checkpoint per model"
log "════════════════════════════════════════════════════════════════"

best_ckpt() {
    local d="$1"
    [[ -d "$d" ]] || return 0
    {
        ls -1 "$d"/ckpt-epoch*-valloss*.ckpt 2>/dev/null
        ls -1 "$d"/default/ckpt-epoch*-valloss*.ckpt 2>/dev/null
    } | awk -F'valloss' '{print $2, $0}' \
      | sort -n | head -1 | awk '{print $2}'
}

# Fallback: if no epoch-monitored ckpt exists (because no full epoch ever
# completed), pick the latest step-snapshot — it's the best we have.
latest_step_ckpt() {
    local d="$1"
    [[ -d "$d" ]] || return 0
    {
        ls -t "$d"/last.ckpt "$d"/default/last.ckpt 2>/dev/null
        ls -t "$d"/ckpt-step*.ckpt "$d"/default/ckpt-step*.ckpt 2>/dev/null
    } | head -1
}

declare -A CKPT
for model in v2 v5; do
    c=$(best_ckpt "$CKPT_ROOT/spy_$model")
    if [[ -z "$c" ]]; then
        c=$(latest_step_ckpt "$CKPT_ROOT/spy_$model/default")
        if [[ -n "$c" ]]; then
            log "  spy_$model: no epoch-monitored ckpt — using latest step snapshot: $c"
        fi
    fi
    if [[ -z "$c" ]]; then
        log "  ✗ spy_$model: NO USABLE CHECKPOINT"
        continue
    fi
    CKPT[$model]="$c"
    log "  ✓ spy_$model: $c"
done

if [[ ${#CKPT[@]} -eq 0 ]]; then
    log "FATAL: zero usable checkpoints after $((N_CHUNKS * 2)) chained jobs — investigate logs"
    exit 1
fi

log ""
log "════════════════════════════════════════════════════════════════"
log "Hand off to standard SPY post-training pipeline"
log "════════════════════════════════════════════════════════════════"
log "EXEC: ./scripts/utils/run_spy_post_training.sh"
exec ./scripts/utils/run_spy_post_training.sh
