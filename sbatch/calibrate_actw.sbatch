#!/bin/bash
#SBATCH --job-name=calib-actw
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Measure the TEACHER's own realised-vs-commanded action error so err_ref stops
# being a guess. err_ref = the teacher's median => a student that follows
# commands as well as the teacher sits at w = 1 + alpha, and only WORSE-than-
# teacher chains are pushed above it. Also tells us empirically whether the
# CoTracker readout works on these 18-frame chains at all.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CA_N=40 python utils/calibrate_actw.py
echo "CALIB-ACTW done $(date)"
