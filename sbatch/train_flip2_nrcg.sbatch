#!/bin/bash
#SBATCH --job-name=flip2-nrcg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
ARM=nrcg bash /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/train_flip2_arms_cde.sh
