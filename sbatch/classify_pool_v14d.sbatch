#!/bin/bash
#SBATCH --job-name=classify-pool-v14d
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=12
#SBATCH --mem=120G
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# CPU-ONLY (no --gpus, non-exclusive -> packs onto a mix node fast).
# Classify EVERY window in the v14d curated pool by z-action direction
# (new turn def: |z2|>0.5|z7|) and dump per-window classes for building
# a direction+city balanced 3000-window pool. 8 CPU tasks shard the pool;
# the ss_vae runs on CPU (tiny). device auto-detects CPU when no GPU.
# =====================================================================
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export OMP_NUM_THREADS=12

TMP=/scratch/u6ex/as1748.u6ex/ARRWM/.tmp/classify_pool
mkdir -p "$TMP"

srun --ntasks-per-node=4 bash -c '
  GPU_RANK=$SLURM_PROCID NUM_GPUS=$SLURM_NTASKS python gen_lmdb.py \
    --analyze_actions \
    --analyze_dump '"$TMP"'/pool_classes \
    --config   configs/causal_lora_diffusion_teacher_v14d.yaml \
    --manifest logs/v14d_critic8_natural/.ride_manifest.pt \
    --ckpt     logs/v14d_critic8_natural/causal_lora_step0002150.pt \
    --lmdb_root    '"$TMP"'/clean \
    --lmdb_cf_root '"$TMP"'/cf \
    --curated_pool paper_assets/v14d_train_windows.json \
    --num_windows  100000 \
    --clean_only'

echo "=== classify-pool done $(date); merging dumps ==="
python utils/build_balanced_pool.py \
  --dumps_glob "$TMP/pool_classes.*.json" \
  --orig_pool  paper_assets/v14d_train_windows.json \
  --report_only
