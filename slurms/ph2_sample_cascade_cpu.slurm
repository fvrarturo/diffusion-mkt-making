#!/bin/bash
#SBATCH --job-name=diffmm_ph2_cascade
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --array=0-399
#SBATCH --output=logs/ph2_cascade_%A_%a.out
#SBATCH --error=logs/ph2_cascade_%A_%a.err

# Memory: 8G — same as regular sampler. Cascade with @torch.no_grad() on both
# ddim_sample and ddim_sample_cascade fits comfortably; real peak ~2-3G per
# task. (Earlier OOMs at 8G/16G/32G were caused by the missing @torch.no_grad
# decorator on ddim_sample_cascade, which built a 200-step computational
# graph that consumed 16G+ during Stage 2 — fixed 2026-05-09.)
#
# Time: 1h covers cascade tasks (~12 min compute + slurm scheduling latency).

# Phase F Model A — cascade sampling (Stage 1 → Stage 2).
#
# Stage 1 = v2-style generator producing book features
# Stage 2 = v10 generator trained with mid_return-only noise/loss masks
#
# Required CLI args via $@ (Hydra style):
#   +cascade_stage1_ckpt=<path>    e.g. v2's best checkpoint
#   +cascade_stage2_ckpt=<path>    e.g. v10's best checkpoint
#   data=intc_top
#   generator=ddim_sample
#   cluster.synthetic_root=<output dir>
#
# Example:
#   sbatch --array=0-399 slurms/ph2_sample_cascade_cpu.slurm \
#       data=intc_top \
#       generator=ddim_sample \
#       +cascade_stage1_ckpt=$HOME/orcd/scratch/diffmm/checkpoints/intc_v2/ckpt-best.ckpt \
#       +cascade_stage2_ckpt=$HOME/orcd/scratch/diffmm/checkpoints/intc_v10/ckpt-best.ckpt \
#       cluster.synthetic_root=$HOME/orcd/scratch/diffmm/synthetic/intc_v10_cascade

source slurms/_common.sh

# Cap PyTorch / MKL / OMP threads to match --cpus-per-task. Without this,
# pytorch defaults to one thread per CPU on the NODE (not per task), causing
# each cascade task to spin up ~64 threads and allocate ~64 per-thread arenas
# → 16G+ memory consumption from threading overhead, not from actual tensors.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2
export NUMEXPR_NUM_THREADS=2

mkdir -p data/manifests
MANIFEST=${MANIFEST:-data/manifests/ph2_sample.json}
if [[ ! -f "$MANIFEST" ]]; then
    echo "FATAL: manifest $MANIFEST not found." >&2
    exit 1
fi

python scripts/13_sample_cascade.py \
    cluster=engaging \
    cluster.device=cpu \
    +task_id="${SLURM_ARRAY_TASK_ID}" \
    +manifest="$MANIFEST" \
    "$@"

echo "=== ph2_cascade task ${SLURM_ARRAY_TASK_ID} done ==="
