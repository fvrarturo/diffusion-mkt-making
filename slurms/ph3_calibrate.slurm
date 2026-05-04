#!/bin/bash
#SBATCH --job-name=diffmm_ph3_cal
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ph3_calibrate_%j.out
#SBATCH --error=logs/ph3_calibrate_%j.err

# Phase 3 §3.2 — calibrate AS-family agents (γ, κ, σ, β_OFI, α, τ_VPIN)
# from the training tapes for one ticker.
#
# Cheap (~30s of compute) since it just streams 60 Parquets and computes
# realized vol + arrival-rate proxy. 64 GB is over-spec — tapes peak at ~2 GB
# in memory — but headroom never hurts.
#
# Usage:
#   sbatch slurms/ph3_calibrate.slurm data=intc_top
#   sbatch slurms/ph3_calibrate.slurm data=tsla_top
#   sbatch slurms/ph3_calibrate.slurm data=intc_top \
#       +calibration.risk_aversion=0.05 +calibration.beta_ofi=2.0

source slurms/_common.sh

python scripts/20_calibrate_agents.py \
    cluster=engaging \
    "$@"

echo "=== ph3_calibrate done ==="
