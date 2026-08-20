#!/bin/bash
#SBATCH --job-name=viz-watcher10k
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=16:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Per-checkpoint viz for the 10k campaign (gated afterok on the smoke, so it
# starts alongside the 4 arms). 4 concurrent per-arm workers on 8 GPUs.
bash /scratch/u6ex/as1748.u6ex/ARRWM/utils/ckpt_viz_watcher10k.sh 15
echo "VIZ-WATCHER10K done $(date)"
