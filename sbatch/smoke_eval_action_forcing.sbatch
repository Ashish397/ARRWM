#!/bin/bash
#SBATCH --job-name=af-smoke-eval
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
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
export OMP_NUM_THREADS=4

echo "=== $(date) host=$(hostname) job=$SLURM_JOB_ID ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

python3 action-forcing/bin/smoke_eval.py \
    --ckpt /scratch/u6ex/as1748.u6ex/ARRWM/logs/v14_balanced_weunz/causal_lora_step0006600.pt \
    --config configs/action_ode_distill.yaml \
    --pair-index 0 \
    --output /scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_eval_fullrank

echo "=== $(date) smoke_eval done ==="
