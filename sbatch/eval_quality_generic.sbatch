#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Visual-quality eval (PSNR/SSIM/LPIPS + indicative FVD) on one checkpoint.
#   $1 = config   $2 = logdir   $3 = out json
#   $4 = windows per ride (optional, default 12)
#   $5 = explicit checkpoint (optional)
# =====================================================================
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG="$1"; LOGDIR="$2"; OUT="$3"; WINDOWS="${4:-12}"; CKPT="${5:-}"
set --   # clear positional args before sourcing conda activate

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

echo "Quality eval  config=$CONFIG  logdir=$LOGDIR  out=$OUT  windows=$WINDOWS  job=$SLURM_JOB_ID  $(date)"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader || true

python utils/eval_quality.py \
  --config "$CONFIG" \
  --logdir "$LOGDIR" \
  --windows "$WINDOWS" \
  ${CKPT:+--checkpoint "$CKPT"} \
  --out "$OUT"

echo "Done on $(date)"
