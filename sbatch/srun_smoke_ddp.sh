#!/bin/bash
# 2-GPU DDP smoke. Exercises world_size>1 paths: no_sync() one-step
# lookahead, gradient-uniformity check, PerSlotRideBatcher lockstep.
#
# Examples:
#   # 4x1 default (bit-identical to legacy path):
#   srun --nodes=1 --gres=gpu:2 --time=01:00:00 --pty bash ./srun_smoke_ddp.sh
#
#   # 2x1 (2 slots, 1 pass/step; 2-step ODE schedule):
#   EXTRA_OVR='num_live_slots=2 passes_per_step=1 action_decay_per_slot=[1.0,0.5] denoising_step_list=[1000,500]' \
#     srun --nodes=1 --gres=gpu:2 --time=01:00:00 --pty bash ./srun_smoke_ddp.sh
#
#   # 2x2 (idea 2/3: 2 slots, 2 passes/step; 4-step ODE schedule):
#   EXTRA_OVR='num_live_slots=2 passes_per_step=2 action_decay_per_slot=[1.0,0.5] denoising_step_list=[1000,500]' \
#     srun --nodes=1 --gres=gpu:2 --time=01:00:00 --pty bash ./srun_smoke_ddp.sh
#
#   # 2x2 with DMD scoring only on slot 0 (idea 2 cost reduction):
#   EXTRA_OVR='num_live_slots=2 passes_per_step=2 action_decay_per_slot=[1.0,0.5] denoising_step_list=[1000,500] dmd_active_slot_policy=slot0' \
#     srun --nodes=1 --gres=gpu:2 --time=01:00:00 --pty bash ./srun_smoke_ddp.sh
set -eo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN

CONFIG=${CONFIG:-configs/longlive_phase1_rolling_staircase.yaml}
LOGDIR=${LOGDIR:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_ddp}
EXTRA_OVR=${EXTRA_OVR:-}
MAX_STEPS=${MAX_STEPS:-2}
ROLLING_STEPS=${ROLLING_STEPS:-2}
SLOTS_PER_RANK=${SLOTS_PER_RANK:-4}

mkdir -p "$LOGDIR"

MASTER_PORT=$((29500 + (SLURM_JOB_ID % 16000) + 3))

torchrun \
  --standalone --nnodes=1 --nproc_per_node=2 \
  --master_port=$MASTER_PORT \
  trainer/causal_rolling_staircase_train.py \
  --config $CONFIG \
  --override \
    log_dir=$LOGDIR \
    wandb_dir=wandb \
    disable_wandb=true \
    slots_per_rank=$SLOTS_PER_RANK \
    rolling_steps_per_iter=$ROLLING_STEPS \
    max_steps=$MAX_STEPS \
    ckpt_interval=999999 \
    log_interval=1 \
    vis_wallclock_seconds=-1 \
    vis_emit_on_first_step=false \
    max_rides=8 \
    min_ride_frames=32 \
    sort_rides_by_length=none \
    $EXTRA_OVR
