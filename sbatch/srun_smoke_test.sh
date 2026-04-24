#!/bin/bash
# Smoke test for phase-1 rolling-staircase training on a single GPU.
# Intended to be launched via `srun --jobid=<existing> --overlap` on the
# already-allocated interactive session. Runs 3 optimizer steps to
# exercise:
#   - warmup + transition (covered on step 0)
#   - live forward → slot-0 commit → trim (mutates KV cache)
#   - backward with gradient checkpointing (recompute vs. live save)
#
# If this runs without raising `CheckpointError`, the KV-cache-index
# snapshot fix in wan/modules/causal_model.py is validated.

set -eo pipefail
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

LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_test
mkdir -p "$LOGDIR"

echo "=== GPU ==="
nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader
echo

MASTER_PORT=$((29500 + (SLURM_JOB_ID % 16000) + 1))

# Single-rank, single-slot smoke with a short-horizon run.
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_port=$MASTER_PORT \
  trainer/causal_rolling_staircase_train.py \
  --config configs/longlive_phase1_rolling_staircase.yaml \
  --override \
    log_dir=$LOGDIR \
    wandb_dir=wandb \
    disable_wandb=true \
    slots_per_rank=1 \
    rolling_steps_per_iter=2 \
    max_steps=3 \
    ckpt_interval=999999 \
    log_interval=1 \
    vis_wallclock_seconds=-1 \
    vis_emit_on_first_step=false \
    max_rides=8 \
    min_ride_frames=32 \
    sort_rides_by_length=none
