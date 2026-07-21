#!/bin/bash
#SBATCH --job-name=iqa-14e-refresh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
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
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
# the 5 token runs the user asked for (excludes v14d + noatok)
export IQA_RUNS="pca8_8node,16node,4node,pca4,pca2,noatok"
export IQA_OUT="analysis/iqa_14e_all.csv"
echo "IQA refresh start $(date) | RUNS=$IQA_RUNS OUT=$IQA_OUT"
python utils/iqa_control_videos.py
echo "IQA refresh done $(date)"
