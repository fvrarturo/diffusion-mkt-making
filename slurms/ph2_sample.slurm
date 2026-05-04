#!/bin/bash
#SBATCH --job-name=diffmm_ph2_sample
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --array=0-399%20
#SBATCH --output=logs/ph2_sample_%A_%a.out
#SBATCH --error=logs/ph2_sample_%A_%a.err

# Synthetic-day sampling. Manifest enumerates (regime, seed) per array task:
#   python scripts/utils/make_manifest.py --kind sample --out data/manifests/ph2_sample.json
#
# Override the checkpoint path on the CLI:
#   sbatch slurms/ph2_sample.slurm generator.checkpoint_path=/path/to/best.ckpt

source slurms/_common.sh

mkdir -p data/manifests
MANIFEST=${MANIFEST:-data/manifests/ph2_sample.json}
if [[ ! -f "$MANIFEST" ]]; then
    echo "FATAL: manifest $MANIFEST not found. Run scripts/utils/make_manifest.py first." >&2
    exit 1
fi

python scripts/13_sample_synthetic.py \
    cluster=engaging \
    +task_id="${SLURM_ARRAY_TASK_ID}" \
    +manifest="$MANIFEST" \
    "$@"

echo "=== ph2_sample task ${SLURM_ARRAY_TASK_ID} done ==="
