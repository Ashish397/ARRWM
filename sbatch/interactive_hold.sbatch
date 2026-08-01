#!/bin/bash
# Held allocation. Does nothing but stay alive so work can be pushed into it
# with `srun --overlap --jobid=<id> <cmd>`, bypassing the queue entirely.
#
#   sbatch --job-name=hold_gpu sbatch/interactive_hold.sbatch
#   srun --overlap --jobid=<id> --ntasks=1 bash -lc '<command>'
#
# HOLD_SECONDS controls the lifetime (default 8h, under the --time below).
#SBATCH --job-name=hold
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/hold_%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/hold_%x_%j.err

echo "HOLD READY jobid=$SLURM_JOB_ID nodes=$SLURM_NNODES gpus=$SLURM_GPUS_ON_NODE"
echo "nodelist=$SLURM_JOB_NODELIST"
echo "started=$(date -u +%Y-%m-%dT%H:%M:%S)"
echo "use: srun --overlap --jobid=$SLURM_JOB_ID --ntasks=1 bash -lc '<cmd>'"
sleep "${HOLD_SECONDS:-28800}"
echo "HOLD EXPIRED $(date -u +%H:%M:%S)"
