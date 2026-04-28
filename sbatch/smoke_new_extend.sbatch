#!/bin/bash
#SBATCH --job-name=smoke_new_extend
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=2
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_new_extend/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_new_extend/%x_%j.err

# =====================================================================
# Smoke for Phase-1 Action-Forcing slide-and-train + v14-aligned
# motion/latent dataset loader.
#
# Validates:
#   * dataset chunk-grain alignment (one z per 3-latent chunk; head
#     latent dropped; n_latent_frames capped at min(zarr-1,
#     n_motion_chunks*3))
#   * _pick_motion_aware_offset npb-snap (offset s % npb == 0)
#   * slide-and-train helper (per-rank MAE, MAX-reduce coordination,
#     synthetic gen+critic backward on untrained ranks)
#   * cmd_actions log line on every rank for verifying alignment
#
# Dataset: 6-ride hi+lo motion symlink pool at
#   logs/smoke_new_extend/encoded_combined/  (created by previous
#   step — check existence; recreate if missing).
#
# Run name + wandb project: smoke_new_extend / longlive-phase1-smoke
# under ashish397-university-of-exeter.
# =====================================================================

set -eo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_new_extend
COMBINED="$LOGDIR/encoded_combined"
mkdir -p "$LOGDIR"

# Recreate the 6-ride symlink pool if it's missing (3 hi-motion + 3
# low-motion). Idempotent — safe if it already exists.
if [ ! -d "$COMBINED" ] || [ -z "$(ls -A "$COMBINED" 2>/dev/null)" ]; then
    mkdir -p "$COMBINED"
    rm -f "$COMBINED"/*.zarr
    for src in /projects/u6ex/fbots/frodobots_encoded_smoke_high/*.zarr \
               /projects/u6ex/fbots/frodobots_encoded_smoke_low/*.zarr; do
        ln -s "$src" "$COMBINED/$(basename "$src")"
    done
fi

# Clear previous smoke artifacts so the run starts clean. Always remove
# the nested run dir (auto_resume=false will skip checkpoint reload, but
# the dir holds stale eval videos).
rm -rf "$LOGDIR/smoke_new_extend"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_action_forcing_train.py \
  --config configs/action_forcing_phase1_freeze.yaml \
  --override \
    encoded_root=$COMBINED \
    motion_skip_dead_rides=false \
    auto_resume=false \
    dmd_context=GT \
    max_steps=10 \
    log_interval=1 \
    checkpoint_interval=999 \
    sample_interval=1 \
    log_dir=$LOGDIR \
    wandb_project=longlive-phase1-smoke \
    run_name=smoke_new_extend
