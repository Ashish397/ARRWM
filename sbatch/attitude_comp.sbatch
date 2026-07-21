#!/bin/bash
#SBATCH --job-name=att-comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Attitude (pitch/roll drift) on all comparator eval videos (6 x 256), resume-safe.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
AD_RUNS=minwm:astra:matrixgame:worldcam:worldplay:yume AD_WINDOWS=ALL \
  AD_OUT=/scratch/u6ex/as1748.u6ex/ARRWM/analysis/eval_final/attitude_comp.csv \
  python utils/attitude_drift.py
echo "ATT-COMP done $(date)"
