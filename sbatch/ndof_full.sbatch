#!/bin/bash
#SBATCH --job-name=ndof-full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export TMPDIR=/tmp TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$USER TRITON_CACHE_DIR=/tmp/triton_$USER
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export NDOF_STRIDE=200 NDOF_OUT=${NDOF_OUT:-analysis/ndof_following.csv}
echo "NDOF full start $(date) | stride=$NDOF_STRIDE"
python utils/ndof_following.py
echo "NDOF full done $(date)"
