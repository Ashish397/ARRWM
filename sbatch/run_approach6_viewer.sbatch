#!/bin/bash
#SBATCH --job-name=approach6-viewer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SAMPLE_IDX=${SAMPLE_IDX:-0}

echo "=== approach6_viewer: sample_idx=${SAMPLE_IDX} ==="

python noise_model/approach6_viewer.py \
    --sample_idx "${SAMPLE_IDX}"

echo "=== Done ==="
ls -lh vis/approach6_*.mp4
