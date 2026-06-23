#!/bin/bash
#SBATCH --job-name=analyze-actions-v14d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:40:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Action-distribution analysis for the 3000 clean windows that the v14d
# gen would select. Encodes z-actions only (no teacher pipe, no gen) and
# prints fwd/bwd/left/right/stationary breakdown (per-frame + per-window).
# NUM_GPUS=1 so this single rank sees ALL 3000 windows.
# =====================================================================
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

TMP=/scratch/u6ex/as1748.u6ex/ARRWM/.tmp/analyze_actions
mkdir -p "$TMP"

GPU_RANK=0 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 python gen_lmdb.py \
  --analyze_actions \
  --config   configs/causal_lora_diffusion_teacher_v14d.yaml \
  --manifest logs/v14d_critic8_natural/.ride_manifest.pt \
  --ckpt     logs/v14d_critic8_natural/causal_lora_step0002150.pt \
  --lmdb_root    "$TMP/clean" \
  --lmdb_cf_root "$TMP/cf" \
  --curated_pool paper_assets/v14d_train_windows.json \
  --num_windows  3000 \
  --clean_only

echo "=== analyze done $(date) ==="
