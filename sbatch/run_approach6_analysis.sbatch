#!/bin/bash
#SBATCH --job-name=approach6-analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=0
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache

MAX_WINDOWS=${MAX_WINDOWS:-99999}
SAVE_SAMPLES=${SAVE_SAMPLES:-20}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-100}

echo "=== approach6_freq_analysis: max_windows=${MAX_WINDOWS} save_samples=${SAVE_SAMPLES} checkpoint_every=${CHECKPOINT_EVERY} ==="

python -u noise_model/approach6_freq_analysis.py \
    --max_windows "${MAX_WINDOWS}" \
    --save_samples "${SAVE_SAMPLES}" \
    --checkpoint_every "${CHECKPOINT_EVERY}"

echo "=== Done ==="
ls -lh noise_model/secant_out/approach6_*
