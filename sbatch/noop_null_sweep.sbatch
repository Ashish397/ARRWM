#!/bin/bash
#SBATCH --job-name=noop-nullsweep
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Sweep the null-action vector: regenerate 6 scenes per candidate z, then
# measure how still each held (BootsTAPIR -> RANSAC affine) plus the fitted
# expansion, which gives the DIRECTION of any residual drift.
#
# z=(0,0) is not zero motion -- the PCA encoder is mean-centred, so z=0 asks
# for the dataset-mean motion, which expands radially (+0.147) = forward creep.
#
# Safe to run concurrently with the same sweep in an interactive allocation:
# inject_eval skips any rollout whose mp4 already exists.
set -e -o pipefail
bash /scratch/u6ex/as1748.u6ex/ARRWM/analysis/testbench_v2/null_sweep.sh
echo "noop-nullsweep done $(date)"
