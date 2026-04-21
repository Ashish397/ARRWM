#!/bin/bash
#SBATCH --job-name=weu-window-viewer
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

# Override these via --export when submitting, e.g.:
#   sbatch --export=ALL,CITY=Madrid,WINDOW_IDX=3 sbatch/run_weu_window_viewer.sbatch
CITY=${CITY:-Madrid}
WINDOW_IDX=${WINDOW_IDX:-0}
NOISE_ALPHA=${NOISE_ALPHA:-1.0}

echo "=== weu_window_viewer: city=${CITY}  window_idx=${WINDOW_IDX}  noise_alpha=${NOISE_ALPHA} ==="

python noise_model/weu_window_viewer.py \
    --city "${CITY}" \
    --window_idx "${WINDOW_IDX}" \
    --noise_alpha "${NOISE_ALPHA}"

echo "=== Done ==="
ls -lh vis/weu_${CITY,,}_$(printf "%04d" ${WINDOW_IDX}).mp4
