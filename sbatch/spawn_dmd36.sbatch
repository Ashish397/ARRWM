#!/bin/bash
#SBATCH --job-name=spawn-dmd36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
export SP_VIDS="dmd36_r08_F=$PWD/logs/eval_final/minwm_smoke/minwm_dmd36_r08_F.mp4:dmd36_r08_Rtrue=$PWD/logs/eval_final/minwm_smoke/minwm_dmd36_r08_L.mp4:dmd36_r01_F=$PWD/logs/eval_final/minwm_smoke/minwm_dmd36_r01_F.mp4:dmd36_r01_Rtrue=$PWD/logs/eval_final/minwm_smoke/minwm_dmd36_r01_L.mp4"
export SP_OUT=$PWD/analysis/eval_final/spawn_dmd36.csv
PYTHONPATH=$PWD python utils/spawn_test.py
echo "SPAWN-DMD36 done"
