#!/bin/bash
#SBATCH --job-name=diffmm_ph1_split
#SBATCH --partition=mit_normal
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/ph1_split_%j.out
#SBATCH --error=logs/ph1_split_%j.err

# Stage A0: split one TAQ gzip into per-date Parquet files. Submit 5 of these
# in parallel (one per gzip):
#
#   sbatch slurms/ph1_split_raw.slurm +gz=data/raw/order_INTC.csv.gz +kind=quotes +ticker=INTC
#   sbatch slurms/ph1_split_raw.slurm +gz=data/raw/order_TSLA.csv.gz +kind=quotes +ticker=TSLA
#   sbatch slurms/ph1_split_raw.slurm +gz=data/raw/trade_INTC.csv.gz +kind=trades +ticker=INTC
#   sbatch slurms/ph1_split_raw.slurm +gz=data/raw/trade_TSLA.csv.gz +kind=trades +ticker=TSLA
#   sbatch slurms/ph1_split_raw.slurm +gz=data/raw/trade_SPY.csv.gz  +kind=trades +ticker=SPY

source slurms/_common.sh

python scripts/01_split_raw_to_parquet.py \
    cluster=engaging \
    "$@"

echo "=== ph1_split done ==="
