#!/bin/bash
# Submit-and-exit: 4-chunk chained training per model, both models in parallel.
# 8 sbatch jobs total, each chunk waits for the previous via --dependency=afterany.
# No polling, no auto-handoff — just submit, print JIDs, exit.
#
# Run on a login node with working scratch mount (e.g. orcd-login001):
#   ./scripts/utils/submit_spy_chain.sh
#
# Then check progress with: squeue -u $USER

set -euo pipefail

CKPT_ROOT="$HOME/orcd/scratch/diffmm/checkpoints"
N_CHUNKS=4

declare -A PRED_TYPE=( [v2]=eps [v5]=v          )
declare -A COND_TYPE=( [v2]=film [v5]=adaln_zero )

[[ -d "$CKPT_ROOT" ]] || {
    echo "FATAL: $CKPT_ROOT not visible — switch to a login node with working scratch (e.g. login001)" >&2
    exit 1
}
mkdir -p logs

submit_chain() {
    local model="$1" pred="$2" cond="$3"
    local ckpt_dir="$CKPT_ROOT/spy_$model"
    local prev=""
    for chunk in $(seq 1 $N_CHUNKS); do
        # Resume logic runs INSIDE the SLURM job at runtime so chunks 2+ pick
        # up whatever checkpoint actually exists when the previous job ends.
        if [[ $chunk -eq 1 ]]; then
            resume_logic='RESUME=""'
            dep=""
        else
            resume_logic='LATEST=$(ls -t '"$ckpt_dir"'/default/last.ckpt '"$ckpt_dir"'/default/ckpt-step*.ckpt '"$ckpt_dir"'/default/ckpt-epoch*.ckpt 2>/dev/null | head -1); [[ -z "$LATEST" ]] && { echo "FATAL: no ckpt to resume" >&2; exit 1; }; echo "resuming from $LATEST"; RESUME="+resume_from=$LATEST"'
            dep="--dependency=afterany:$prev"
        fi
        local jid
        jid=$(sbatch --parsable $dep \
            --partition=mit_normal_gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G --time=06:00:00 \
            --job-name="spy_${model}_c${chunk}" \
            --output="logs/spy_${model}_c${chunk}_%j.out" \
            --error="logs/spy_${model}_c${chunk}_%j.err" \
            --wrap="source slurms/_common.sh 2>/dev/null || true; cd $PWD; $resume_logic; \
                python scripts/11_finetune_generator.py \
                    cluster=engaging data=spy_top \
                    generator.schedule.prediction_type=$pred \
                    generator.model.conditioning_type=$cond \
                    cluster.checkpoint_root=$ckpt_dir \
                    +generator.checkpointing.every_n_train_steps=10000 \
                    \$RESUME")
        printf "  spy_%s chunk %d: JID=%s%s\n" "$model" "$chunk" "$jid" \
            "$([[ -n "$dep" ]] && echo "  ($dep)")"
        prev="$jid"
    done
}

echo "Submitting 4-chunk chains for spy_v2 and spy_v5 (8 jobs total)…"
for m in v2 v5; do
    echo
    echo "── spy_$m ──"
    submit_chain "$m" "${PRED_TYPE[$m]}" "${COND_TYPE[$m]}"
done

echo
echo "Done. Watch with:  squeue -u \$USER -o '%.18i %.20j %.10T %.10M %R'"
