#!/bin/bash
#SBATCH --job-name=diffmm_ph2_finetune
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=logs/ph2_finetune_%j.out
#SBATCH --error=logs/ph2_finetune_%j.err

# Phase-2 generator fine-tuning. Default partition is mit_normal_gpu (6h cap).
# For runs longer than 6h, sched_mit_kburdge_r8 (4× A100 per node, 14-day cap)
# is preferable IF your account has group access. Override at submit time:
#   sbatch --partition=sched_mit_kburdge_r8 --gres=gpu:a100:1 --time=24:00:00 \
#          slurms/ph2_finetune.slurm ...
# The script supports `+resume_from=<ckpt>` to chain 6h chunks: Lightning
# restores optimizer state, scheduler state, epoch counter, and global_step,
# so chained chunks are indistinguishable from one uninterrupted run.

source slurms/_common.sh

python scripts/11_finetune_generator.py \
    cluster=engaging \
    "$@"

echo "=== ph2_finetune done ==="
