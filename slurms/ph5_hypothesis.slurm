#!/bin/bash
#SBATCH --job-name=diffmm_ph5
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --output=logs/ph5_hypothesis_%j.out
#SBATCH --error=logs/ph5_hypothesis_%j.err

# Phase 5 — open holdout (one-time!) + run central rank-correlation test.
# Cheap CPU job: replays 3 agents on 20 holdout tapes (~5 min) + 1000-bootstrap
# on the resulting Sharpe rankings (~10 sec).
#
# Pre-requisites already on disk:
#   results/agent_params/{ticker}_{A1,A2,A3}.json
#   results/backtests/real/{ticker}/ranking_pi_hist.json
#   results/backtests/synthetic/{ticker}/ranking_pi_diff.json
#
# Outputs:
#   results/backtests/holdout/{ticker}/{metrics.csv, ranking_pi_truth.json}
#   results/hypothesis/{ticker}/{bootstrap.json, figure7.pdf, figure7.png}
#
# Usage:
#   sbatch slurms/ph5_hypothesis.slurm data=intc_top
#   sbatch slurms/ph5_hypothesis.slurm data=tsla_top

source slurms/_common.sh

echo "=== Step 1/2: open holdout, replay agents → π_truth ==="
python scripts/40_open_holdout.py cluster=engaging "$@"

echo
echo "=== Step 2/2: bootstrap rank-correlation test ==="
python scripts/41_hypothesis_test.py cluster=engaging "$@"

echo "=== ph5 hypothesis done ==="
