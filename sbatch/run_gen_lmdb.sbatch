#!/bin/bash
#SBATCH --job-name=gen-lmdb-v14
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=23:30:00
#SBATCH --requeue
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
export NUM_GPUS=4

mkdir -p \
    /projects/u6ex/fbots/frodobots_lmdb/v14 \
    /projects/u6ex/fbots/frodobots_lmdb_counterfac/v14 \
    logs

echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

for GPU in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES=$GPU GPU_RANK=$GPU NUM_GPUS=4 \
        python3 gen_lmdb.py \
        > logs/gen_lmdb_gpu${GPU}_${SLURM_JOB_ID}.log 2>&1 &
done

wait
echo "=== $(date) gen_lmdb finished ==="

echo "Clean LMDB count:     $(ls /projects/u6ex/fbots/frodobots_lmdb/v14 | wc -l)"
echo "Counterfac LMDB count: $(ls /projects/u6ex/fbots/frodobots_lmdb_counterfac/v14 | wc -l)"
