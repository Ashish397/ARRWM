#!/bin/bash
#SBATCH --job-name=v14-action-swap-eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Independent action-controllability eval (counterfactual command swap)
# on the existing v14 checkpoint. Single GPU, inference only.
# Judge = frozen CoTracker + frozen ss_vae on the GENERATED frames.
# =====================================================================

set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "Starting v14 action-swap eval on $(date)  (job $SLURM_JOB_ID)"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader || true

python utils/eval_action_swap.py \
  --config configs/causal_lora_diffusion_teacher_v14.yaml \
  --logdir /scratch/u6ex/as1748.u6ex/ARRWM/logs/v14_balanced_weunz \
  --seeds 3 \
  --out /scratch/u6ex/as1748.u6ex/ARRWM/paper_assets/action_swap_results.json

echo "Done on $(date)"
