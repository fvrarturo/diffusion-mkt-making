#!/bin/bash
# Common scaffolding sourced by every SLURM job in this project.
set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

# Conda env activation (recreated on cluster from environment.yml).
source /etc/profile.d/modules.sh 2>/dev/null || true
module load miniforge 2>/dev/null || module load anaconda3 2>/dev/null || true
if command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
  conda activate diffmm 2>/dev/null || {
    echo "FATAL: conda env 'diffmm' not found. Create it from environment.yml." >&2
    exit 1
  }
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "=== Host: $(hostname)  Date: $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
echo "=== Job: ${SLURM_JOB_ID:-NA}  Array: ${SLURM_ARRAY_TASK_ID:-NA} ==="
echo "=== Python: $(python -c 'import sys; print(sys.executable, sys.version.split()[0])') ==="
echo "=== Git SHA: $(git rev-parse HEAD 2>/dev/null || echo unknown) ==="
echo "=== CWD: $(pwd) ==="
