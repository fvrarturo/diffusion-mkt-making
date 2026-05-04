#!/bin/bash
#SBATCH --job-name=diffmm_ph6_cpu
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=256G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ph6_validate_cpu_%j.out
#SBATCH --error=logs/ph6_validate_cpu_%j.err

# Generator validation — Parts B (marginals), C (temporal), D (conditional).
# Pure CPU stats on already-generated data. The heavy memory request is
# because we concatenate 10 real val tapes (~30M events) and 1000 synth tapes
# into single Polars DataFrames; ACF on a 30M-element series materializes
# multiple numpy copies, and inter-day W-1 over 45 day-pairs holds many
# arrays in flight.
#
# Configure with env vars:
#   SYNTH_DIR  = path to synthetic_v2_stitched/INTC/
#   N_SYNTH    = synth tapes sampled per regime (default 250)
#
# Example:
#   SYNTH_DIR=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v2_stitched/INTC \
#   sbatch slurms/ph6_validate_cpu.slurm

source slurms/_common.sh

: "${SYNTH_DIR:?SYNTH_DIR must be set}"
N_SYNTH="${N_SYNTH:-250}"

python scripts/50_validate_generator.py \
    --parts B C D \
    --real-dir data/processed/val/ \
    --synth-dir "$SYNTH_DIR" \
    --ticker INTC \
    --out-dir results/validation/INTC \
    --n-synth-per-regime "$N_SYNTH"

echo "=== ph6 validate cpu done ==="
