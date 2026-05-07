#!/bin/bash
#SBATCH --job-name=diffmm_ph2_sample_cpu
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --array=0-399%20
#SBATCH --output=logs/ph2_sample_cpu_%A_%a.out
#SBATCH --error=logs/ph2_sample_cpu_%A_%a.err

# CPU variant of ph2_sample.slurm. Use when GPU partition QOS is constrained —
# CPU partition typically has much higher concurrent-task headroom, so even
# though per-task compute is ~3-4× slower (~30s vs ~10s), throughput at
# %20 concurrency beats %2 on GPU by 5-10×.
#
# Override the checkpoint path on the CLI (same as the GPU version):
#   sbatch slurms/ph2_sample_cpu.slurm \
#       generator=ddim_sample \
#       data=intc_top \
#       generator.checkpoint_path=/path/to/best.ckpt \
#       cluster.synthetic_root=/path/to/output

source slurms/_common.sh

mkdir -p data/manifests
MANIFEST=${MANIFEST:-data/manifests/ph2_sample.json}
if [[ ! -f "$MANIFEST" ]]; then
    echo "FATAL: manifest $MANIFEST not found. Run scripts/utils/make_manifest.py first." >&2
    exit 1
fi

python scripts/13_sample_synthetic.py \
    cluster=engaging \
    cluster.device=cpu \
    +task_id="${SLURM_ARRAY_TASK_ID}" \
    +manifest="$MANIFEST" \
    "$@"

echo "=== ph2_sample_cpu task ${SLURM_ARRAY_TASK_ID} done ==="
