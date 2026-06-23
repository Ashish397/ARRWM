#!/bin/bash
#SBATCH --job-name=gen-lmdb-v14d-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:50:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# SMOKE: generate 5 CLEAN ODE pairs with the 14d teacher (no CF), then
# VAE-decode the teacher x0 of each to mp4 in eval/ode_F_smoke for visual
# verification BEFORE committing to the full 3000-window run.
# 1 GPU, GPU_RANK=0/NUM_GPUS=1 so this single task processes all 5 windows.
# =====================================================================
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

SMOKE_CLEAN=/projects/u6ex/fbots/frodobots_lmdb/v14d_smoke_mixed
SMOKE_CF=/projects/u6ex/fbots/frodobots_lmdb_counterfac/v14d_smoke_mixed   # unused (clean_only)
mkdir -p "$SMOKE_CLEAN" "$SMOKE_CF" eval/ode_F_smoke_mixed

echo "=== gen 10 MIXED clean ODE pairs (2/direction, 14d) $(date) ==="
GPU_RANK=0 NUM_GPUS=1 CUDA_VISIBLE_DEVICES=0 python gen_lmdb.py \
  --config   configs/causal_lora_diffusion_teacher_v14d.yaml \
  --manifest logs/v14d_critic8_natural/.ride_manifest.pt \
  --ckpt     logs/v14d_critic8_natural/causal_lora_step0002150.pt \
  --lmdb_root    "$SMOKE_CLEAN" \
  --lmdb_cf_root "$SMOKE_CF" \
  --curated_pool paper_assets/v14d_smoke_mixed.json \
  --num_windows  10 \
  --clean_only

echo "=== decode 10 windows -> eval/ode_F_smoke_mixed $(date) ==="
CUDA_VISIBLE_DEVICES=0 python utils/decode_lmdb_smoke.py \
  --lmdb_dir "$SMOKE_CLEAN" \
  --out_dir  eval/ode_F_smoke_mixed \
  --n 10 --fps 12

echo "=== smoke done $(date) ==="
ls -la eval/ode_F_smoke_mixed/
