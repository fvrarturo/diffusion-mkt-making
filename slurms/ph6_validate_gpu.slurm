#!/bin/bash
#SBATCH --job-name=diffmm_ph6_gpu
#SBATCH --partition=mit_normal_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=logs/ph6_validate_gpu_%j.out
#SBATCH --error=logs/ph6_validate_gpu_%j.err

# Generator validation — Parts E (training diagnostics) + F (model internals).
# GPU job: per-timestep MSE, x0_clip activation, regime PCA, FiLM modulation,
# attention maps, guidance sweep, per-feature MSE.
#
# Memory: 64G is plenty (model is 10M params, val windows are subsampled).
# The heavier cpu-side stats (B/C/D) run separately via ph6_validate_cpu.slurm.
#
# Required env vars:
#   CKPT       = v2 checkpoint
#   SYNTH_DIR  = synthetic_v2_stitched/INTC/  (only needed for B/C/D parts; ignored by E/F)
# Optional:
#   V1_CKPT    = v1 checkpoint (for E.2 comparison)
#   V1_LOG     = v1 training .out log (for E.1)
#   V2_LOG     = v2 training .out log (for E.1)
#   N_SEEDS_SWEEP = guidance sweep seeds (default 20; 50 for the full plan)
#
# Example:
#   CKPT=/orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v2/ckpt-epoch004-valloss0.0106.ckpt \
#   V1_CKPT=/orcd/scratch/orcd/007/favara/diffmm/checkpoints/intc_v1_short/best.ckpt \
#   SYNTH_DIR=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v2_stitched/INTC \
#   V2_LOG=logs/ph2_finetune_12305274.out \
#   sbatch slurms/ph6_validate_gpu.slurm

source slurms/_common.sh

: "${CKPT:?CKPT must be set}"
: "${SYNTH_DIR:?SYNTH_DIR must be set (for the synth-aware parts to read tape regime labels)}"
N_SEEDS_SWEEP="${N_SEEDS_SWEEP:-20}"
PREDICTION_TYPE="${PREDICTION_TYPE:-eps}"        # v4+ Phase B = "v"; v2/v3/v3.5 = "eps"
CONDITIONING_TYPE="${CONDITIONING_TYPE:-film}"   # v5+ Phase C = "adaln_zero"; v2-v4 = "film"
USE_EDM="${USE_EDM:-0}"                          # v6+ Phase D = "1"; v2-v5 = "0"
EDM_SIGMA_DATA="${EDM_SIGMA_DATA:-0.5}"
USE_COPULA="${USE_COPULA:-0}"                    # v8+ Phase E = "1"; v2-v7 = "0"

OPT_FLAGS=""
[[ -n "${V1_CKPT:-}" ]] && OPT_FLAGS="$OPT_FLAGS --v1-checkpoint $V1_CKPT"
[[ -n "${V1_LOG:-}"  ]] && OPT_FLAGS="$OPT_FLAGS --v1-log $V1_LOG"
[[ -n "${V2_LOG:-}"  ]] && OPT_FLAGS="$OPT_FLAGS --v2-log $V2_LOG"
[[ "$USE_EDM" == "1" ]] && OPT_FLAGS="$OPT_FLAGS --use-edm --edm-sigma-data $EDM_SIGMA_DATA"
[[ "$USE_COPULA" == "1" ]] && OPT_FLAGS="$OPT_FLAGS --use-copula"

python scripts/50_validate_generator.py \
    --parts E F \
    --real-dir data/processed/val/ \
    --synth-dir "$SYNTH_DIR" \
    --ticker INTC \
    --out-dir results/validation/INTC \
    --checkpoint "$CKPT" \
    --norm-stats data/processed/norm_stats_INTC.json \
    --n-seeds-sweep "$N_SEEDS_SWEEP" \
    --prediction-type "$PREDICTION_TYPE" \
    --conditioning-type "$CONDITIONING_TYPE" \
    $OPT_FLAGS

echo "=== ph6 validate gpu done ==="
