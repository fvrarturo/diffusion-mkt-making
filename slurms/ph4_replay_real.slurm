#!/bin/bash
#SBATCH --job-name=diffmm_ph4_real
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ph4_replay_real_%j.out
#SBATCH --error=logs/ph4_replay_real_%j.err

# Phase 4 §4.2 — historical replay on val tapes (real data) → π_hist.
#
# Single CPU job: 3 agents × 10 val tapes per ticker = 30 backtest runs.
# Each tape is millions of events; the FillSimulator is a serial Python loop.
# Expected wall-clock: a few minutes per ticker.
#
# Memory budget: each tape decompresses to ~500 MB in Polars + some intraday
# bookkeeping. 128 GB is comfortable safety margin.
#
# Usage:
#   sbatch slurms/ph4_replay_real.slurm data=intc_top
#   sbatch slurms/ph4_replay_real.slurm data=tsla_top
#
# Requires: scripts/20_calibrate_agents.py has already run for this ticker.

source slurms/_common.sh

python scripts/21_replay_real.py \
    cluster=engaging \
    "$@"

echo "=== ph4_replay_real done ==="
