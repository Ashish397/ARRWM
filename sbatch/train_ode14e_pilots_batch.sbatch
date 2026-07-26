#!/bin/bash
#SBATCH --job-name=ode14e-pilots
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# 4 sequential pilot ODE trainings (gt_cap/flip2/dir4/dir8, 250 steps each,
# fresh from the 14e teacher, pinned 20-step rung grid). Skip-guarded so an
# interactive-node run of the same script is harmless.
bash /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/_train_pilots_seq.sh
