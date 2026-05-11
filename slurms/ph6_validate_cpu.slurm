#!/bin/bash
#SBATCH --job-name=diffmm_ph6_cpu
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --output=logs/ph6_validate_cpu_%j.out
#SBATCH --error=logs/ph6_validate_cpu_%j.err

# Generator validation — Parts B (marginals), C (temporal), D (conditional).
# Pure CPU stats on already-generated data.
#
# Resource ask reduced 2026-05-08 to enable parallel batch runs across all
# generators (256G mem → 64G, 8 CPUs → 4, 2h → 45min). Real peak observed
# at ~30-50G for the 30M-event val concat + ACF; 64G has comfortable headroom.
# If you hit OOM with N_SYNTH > 250, raise --mem back to 128G.
#
# Configure with env vars:
#   SYNTH_DIR  = path to synthetic_v2_stitched/INTC/
#   N_SYNTH    = synth tapes sampled per regime (default 250; lower → less memory)
#   OUT_DIR    = output directory (default results/validation/<TICKER>).
#                Override to allow parallel runs writing to distinct dirs.
#   TICKER     = ticker symbol (default INTC). Used both for the --ticker arg
#                and for filtering real parquets in data/processed/val/.
#
# Example:
#   SYNTH_DIR=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v2_stitched/INTC \
#   OUT_DIR=results/validation_v2/INTC \
#   sbatch slurms/ph6_validate_cpu.slurm
#
#   # Cross-ticker run:
#   SYNTH_DIR=$HOME/orcd/scratch/diffmm/synthetic/tsla_v2/TSLA \
#   OUT_DIR=results/_archive/tsla_v2/validation TICKER=TSLA \
#   sbatch slurms/ph6_validate_cpu.slurm

source slurms/_common.sh

: "${SYNTH_DIR:?SYNTH_DIR must be set}"
N_SYNTH="${N_SYNTH:-250}"
TICKER="${TICKER:-INTC}"
OUT_DIR="${OUT_DIR:-results/validation/$TICKER}"

python scripts/50_validate_generator.py \
    --parts B C D \
    --real-dir data/processed/val/ \
    --synth-dir "$SYNTH_DIR" \
    --ticker "$TICKER" \
    --out-dir "$OUT_DIR" \
    --n-synth-per-regime "$N_SYNTH"

echo "=== ph6 validate cpu done ==="
