#!/bin/bash
#SBATCH --job-name=gen-lmdb-v14d
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# ODE-F PREREQ: regenerate the ODE distillation dataset from the FIXED
# v14d teacher on the NATURAL-motion + backward pool (vs the old v14/v14b
# motion_min-0.30 pool that had 0% stationary). gen_lmdb runs the v14d
# teacher's PF-ODE over each window and snapshots the trajectory (clean +
# counterfactual). Sharded across all GPUs (gpu_rank/num_gpus internally).
#   NUM_WINDOWS: tune to the balanced-coverage target (backward is the cap).
# =====================================================================
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8

NUM_WINDOWS="${NUM_WINDOWS:-6000}"   # 6000 CLEAN balanced windows (no CF)

# Per-task sharding: gen_lmdb reads GPU_RANK/NUM_GPUS from env to slice the
# window list (samples[GPU_RANK::NUM_GPUS]); SLURM does NOT set those, so we
# set them per task from SLURM_PROCID/SLURM_NTASKS inside a bash -c wrapper.
# --gpus-per-task=1 already pins one GPU per task (gen_lmdb uses cuda:0).
# --clean_only: no counterfactual pass (the whole budget goes to clean).
srun --ntasks-per-node=4 --gpus-per-task=1 bash -c '
  GPU_RANK=$SLURM_PROCID NUM_GPUS=$SLURM_NTASKS python gen_lmdb.py \
    --config   configs/causal_lora_diffusion_teacher_v14d.yaml \
    --manifest logs/v14d_critic8_natural/.ride_manifest.pt \
    --ckpt     logs/v14d_critic8_natural/causal_lora_step0002150.pt \
    --lmdb_root    /projects/u6ex/fbots/frodobots_lmdb/v14d \
    --lmdb_cf_root /projects/u6ex/fbots/frodobots_lmdb_counterfac/v14d \
    --curated_pool paper_assets/v14d_train_windows_balanced.json \
    --num_windows  '"$NUM_WINDOWS"' \
    --clean_only'

echo "gen_lmdb v14d done $(date)"
