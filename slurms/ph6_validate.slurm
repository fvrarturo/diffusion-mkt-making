#!/bin/bash
# DEPRECATED — split into two jobs to satisfy node-level memory caps:
#   slurms/ph6_validate_cpu.slurm   — Parts B/C/D on mit_normal (256G mem)
#   slurms/ph6_validate_gpu.slurm   — Parts E/F   on mit_normal_gpu (1 GPU + 64G)
#
# CPU stats over 10 real val days × 1000 synth tapes need ~30-50G; the GPU
# parts only need a model forward pass on subsampled windows.

echo "Use slurms/ph6_validate_cpu.slurm and slurms/ph6_validate_gpu.slurm instead."
exit 1
