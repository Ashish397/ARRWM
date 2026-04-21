#!/bin/bash
#SBATCH --job-name=pre-encode-direct
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail

ARRWM=/scratch/u6ex/as1748.u6ex/ARRWM
cd $ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR

SCRIPT=$ARRWM/utils/pre_encode_direct.py
BASE_DIR=/projects/u6ex/fbots/frodobots_raw
OUTPUT=/projects/u6ex/fbots/frodobots_encoded
VAE=/scratch/u6ex/as1748.u6ex/frodobots/Wan2.1-T2V-1.3B/Wan2.1_VAE.pth

for RANK in 0 1 2 3; do
    CSV=$ARRWM/city_encode/direct_rides_rank${RANK}.csv
    CUDA_VISIBLE_DEVICES=$RANK python3 $SCRIPT \
        --rides_csv "$CSV" \
        --base_dir "$BASE_DIR" \
        --output_root "$OUTPUT" \
        --vae_path "$VAE" \
        --device cuda \
        --model_dtype float16 \
        --latent_dtype float16 \
        > $ARRWM/logs/pre_encode_direct_rank${RANK}.log 2>&1 &
done

wait
echo "All 4 GPU tasks completed."
