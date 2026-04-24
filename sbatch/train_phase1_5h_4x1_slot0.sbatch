#!/bin/bash
#SBATCH --job-name=phase1-5h-4x1-slot0
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=5:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Rolling-Staircase DMD, 5h, 1 node × 4 GPUs.
#
# VARIANT: 4 slots × 1 pass/step. DMD gradient is applied to slot 0
# ONLY every rolling step (no random rotation). Action teacher runs on
# slot 0 only. This is the "anchor-only" baseline: slot 0 still
# attends bidirectionally to the other 3 live slots during the
# generator forward, so it benefits from the full live-window
# context, but only slot 0's pred_x0 is pushed through the DMD real +
# fake scorers.
#
#   * num_live_slots         = 4
#   * passes_per_step        = 1
#   * denoising_step_list    = [250,500,750,1000]
#   * action_decay_per_slot  = [1.0, 0.75, 0.5, 0.25]
#   * real_score_num_gt_chunks = 4  (slot 0: 4 GT + kv_anchor_chunk
#       [= most recently committed KV chunk] + live_0 target.)
#   * dmd_active_slot_policy = slot0
#   * action_teacher_mode    = slot0
#
# Matched against the random-slot variant (same YAML overrides except
# dmd_active_slot_policy) so you can directly read off the value of
# spreading gradient vs concentrating it on slot 0.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/longlive_phase1_rolling_staircase.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/phase1_5h_4x1_slot0
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

mkdir -p "$LOGDIR"
mkdir -p logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "Starting phase-1 4x1 slot-0-only DMD (1 node, 4 GPUs, 5h) on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_rolling_staircase_train.py \
  --config $CONFIG \
  --override \
    log_dir=$LOGDIR \
    wandb_dir=$WANDB_SAVE_DIR \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_weu \
    caption_root=/projects/u6ex/fbots/frodobots_captions/train \
    motion_root=/projects/u6ex/fbots/frodobots_motion \
    min_ride_frames=80 \
    slots_per_rank=4 \
    rolling_steps_per_iter=16 \
    num_live_slots=4 \
    passes_per_step=1 \
    "denoising_step_list=[250,500,750,1000]" \
    "action_decay_per_slot=[1.0,0.75,0.5,0.25]" \
    real_score_num_gt_chunks=4 \
    dmd_active_slot_policy=slot0 \
    action_teacher_mode=slot0 \
    aux_loss_slot_policy=slot0

echo "Training completed on $(date)"
