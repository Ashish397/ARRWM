#!/bin/bash
#SBATCH --job-name=gpu-noopint
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Idle GPU allocation for interactively sweeping the null-action vector.
# Attach with:  srun --jobid=<JOBID> --overlap -n1 bash <script>
echo "gpu-noopint holding $(hostname) with 4 GPUs from $(date)"
sleep 14400
