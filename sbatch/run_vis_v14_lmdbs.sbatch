#!/bin/bash
#SBATCH --job-name=vis-v14-lmdbs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=0:45:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export HF_HUB_CACHE=$HF_HOME
export HUGGINGFACE_HUB_CACHE=$HF_HOME
export TRANSFORMERS_CACHE=$HF_HOME
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "=== $(date) host=$(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

V14_CLEAN=/projects/u6ex/fbots/frodobots_lmdb/v14
V14_CF=/projects/u6ex/fbots/frodobots_lmdb_counterfac/v14

echo "=== Visualising v14 CLEAN LMDB ==="
python testing/test_lmdb_sample.py \
    --lmdb_dir  "$V14_CLEAN" \
    --normal_lmdb "$V14_CLEAN" \
    --output_dir vis/lmdb_sample_v14

echo "=== Visualising v14 COUNTERFACTUAL LMDB ==="
python testing/test_lmdb_sample.py \
    --lmdb_dir    "$V14_CF" \
    --normal_lmdb "$V14_CLEAN" \
    --output_dir  vis/lmdb_sample_v14_counterfac

echo "=== $(date) finished ==="
ls -R vis/lmdb_sample_v14 vis/lmdb_sample_v14_counterfac 2>/dev/null | tail -40
