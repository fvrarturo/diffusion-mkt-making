#!/bin/bash
#SBATCH --job-name=diffmm_trades_eval
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --output=logs/trades_eval_%j.out
#SBATCH --error=logs/trades_eval_%j.err

# TRADES supplementary metrics (script 62) + predictive score (script 63)
# over every available generator. Wraps scripts/utils/run_trades_eval_all.sh.
#
# Why a SLURM job (not login node): loading 18 models × 100 tapes × 4 regimes
# = 7200 parquets requires 5-10 GB peak RAM (decoded DataFrames). Login nodes
# OOM-kill at this scale. mit_normal compute nodes have plenty of headroom.
#
# Usage:
#   sbatch slurms/ph7_trades_eval.slurm
#
# Reattach to follow progress (in another terminal):
#   tail -f logs/trades_eval_<JID>.out
#
# Memory: 128G is comfortable headroom (peak observed ~10G during script 63's
# windowed-tensor construction across 18 models). If you hit OOM, raise --mem
# OR reduce --max-tapes / --max-tapes-train in the orchestrator script.
#
# Time: 4h covers stage 1 (~30 min) + stage 2 (~2-3h sequential LSTM training).

source slurms/_common.sh

./scripts/utils/run_trades_eval_all.sh

echo "=== ph7 trades_eval done ==="
