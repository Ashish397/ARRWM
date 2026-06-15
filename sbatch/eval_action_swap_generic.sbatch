#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Parameterized independent action-swap eval (frozen CoTracker+ss_vae judge)
# on a single checkpoint. Single GPU, inference only.
#   $1 = config path
#   $2 = logdir (auto_resume loads the latest checkpoint there)
#   $3 = output json path
# =====================================================================

set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG="$1"
LOGDIR="$2"
OUT="$3"
CKPT="${4:-}"   # optional explicit checkpoint (skips a corrupt auto_resume latest)

# Clear positional args so the sourced conda 'activate' script doesn't inherit
# them ($1/$2/$3 would otherwise be passed to `activate`, which rejects them).
set --

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "Action-swap eval  config=$CONFIG  logdir=$LOGDIR  out=$OUT  job=$SLURM_JOB_ID  $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

python utils/eval_action_swap.py \
  --config "$CONFIG" \
  --logdir "$LOGDIR" \
  ${CKPT:+--checkpoint "$CKPT"} \
  --seeds 3 \
  --out "$OUT"

echo "Done on $(date)"
