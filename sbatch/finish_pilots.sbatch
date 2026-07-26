#!/bin/bash
#SBATCH --job-name=finish-pilots
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Finish whatever pilots remain (skip-guarded + auto-resume), then probe all.
bash /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/_train_pilots_seq.sh
bash /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/_probe_pilots.sh
