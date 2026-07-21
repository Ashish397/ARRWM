#!/bin/bash
#SBATCH --job-name=iqa-14e
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}
export TRITON_CACHE_DIR=/tmp/triton_${USER}
export IQA_OUT=${IQA_OUT:-analysis/iqa_14e.csv}
export IQA_LIMIT=${IQA_LIMIT:-0}
echo "IQA start $(date) | LIMIT=$IQA_LIMIT OUT=$IQA_OUT"
python utils/iqa_control_videos.py
echo "IQA done $(date)"
