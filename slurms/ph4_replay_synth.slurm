#!/bin/bash
#SBATCH --job-name=diffmm_ph4_synth
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=logs/ph4_replay_synth_%j.out
#SBATCH --error=logs/ph4_replay_synth_%j.err

# Phase 4 §4.3 — synthetic stress replay → π_diff.
#
# 3 agents × 4 regimes × 1600 synthetic tapes = 19,200 backtest runs per
# ticker. Each synthetic tape is tiny (256 events ≈ 12 KB on disk), loaded
# one-at-a-time. Observed wall-clock on the v2 INTC run was ~1 minute.
#
# Memory: < 1 GB peak in practice; 32 GB is comfortable headroom and lets
# SLURM schedule on any mit_normal node immediately.
#
# Usage:
#   # Default — uses cluster.synthetic_root from configs/cluster/engaging.yaml
#   sbatch slurms/ph4_replay_synth.slurm data=intc_top
#
#   # Override the synthetic root explicitly (e.g. v2 outputs):
#   sbatch slurms/ph4_replay_synth.slurm data=intc_top \
#       +synthetic_root=/orcd/scratch/orcd/007/favara/diffmm/synthetic_v2
#
# Requires: scripts/20_calibrate_agents.py has already run for this ticker.

source slurms/_common.sh

python scripts/22_replay_synthetic.py \
    cluster=engaging \
    "$@"

echo "=== ph4_replay_synth done ==="
