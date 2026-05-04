#!/bin/bash
#SBATCH --job-name=diffmm_ph1_finalize
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ph1_finalize_%j.out
#SBATCH --error=logs/ph1_finalize_%j.err

# Stage A2: splits + quantiles + canonical Parquets. Run once per ticker.
# Cheap to re-run when split sizes change.
#
#   sbatch slurms/ph1_finalize.slurm data=intc_top
#   sbatch slurms/ph1_finalize.slurm data=tsla_top
#
# Override sizes inline:
#   sbatch slurms/ph1_finalize.slurm data=intc_top data.splits.n_train_days=80

source slurms/_common.sh

python scripts/03_finalize_tapes.py \
    cluster=engaging \
    "$@"

echo "=== ph1_finalize done ==="
