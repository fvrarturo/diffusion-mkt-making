#!/bin/bash
#SBATCH --job-name=diffmm_ph1_clean
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --array=0-503%50
#SBATCH --output=logs/ph1_clean_%A_%a.out
#SBATCH --error=logs/ph1_clean_%A_%a.err

# Stage A1: per-date cleaning array job. Generate the manifest first:
#
#   python scripts/utils/make_manifest.py --kind clean \
#       --out data/manifests/ph1_clean.json --tickers INTC TSLA
#
# Then update the --array range above to match (manifest length - 1).
# Default 504 = 252 trading days × 2 tickers.

source slurms/_common.sh

MANIFEST=${MANIFEST:-data/manifests/ph1_clean.json}
if [[ ! -f "$MANIFEST" ]]; then
    echo "FATAL: manifest $MANIFEST not found." >&2
    echo "Generate it with: python scripts/utils/make_manifest.py --kind clean --out $MANIFEST" >&2
    exit 1
fi

# Determine which data config to load based on the ticker for this task.
TICKER=$(python -c "import json; m=json.load(open('$MANIFEST')); print(m[$SLURM_ARRAY_TASK_ID]['ticker'])")
DATA_CFG=$(echo "$TICKER" | tr '[:upper:]' '[:lower:]')_top

python scripts/02_build_tapes.py \
    cluster=engaging \
    data="$DATA_CFG" \
    +task_id="${SLURM_ARRAY_TASK_ID}" \
    +manifest="$MANIFEST" \
    "$@"

echo "=== ph1_clean task ${SLURM_ARRAY_TASK_ID} done ==="
