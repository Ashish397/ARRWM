#!/bin/bash
#SBATCH --job-name=viz-watcherdmd
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Per-checkpoint viz for the DMD-10k arms (ckpt every 25 steps).
bash /scratch/u6ex/as1748.u6ex/ARRWM/utils/ckpt_viz_watcherdmd.sh 11
echo "VIZ-WATCHERDMD done $(date)"
