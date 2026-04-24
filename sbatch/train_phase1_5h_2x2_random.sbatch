#!/bin/bash
#SBATCH --job-name=phase1-5h-2x2-random
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
# VARIANT: 2 slots × 2 passes/step. Each rolling step:
#   * runs 2 internal denoising passes (pass-0 input ladder [500,1000];
#     pass-1 input ladder [250, 750], derived from the step budget).
#   * picks ONE random PASS (DDP-synced) to carry the autograd graph;
#     the other pass runs under no_grad.
#   * within that grad-pass, the DMD scorer forwards only ONE RANDOM
#     SLOT (dmd_active_slot_policy=random_any, dmd_num_active_slots=1,
#     DDP-synced).
#   * Net result: gradient flows through exactly ONE (pass, slot)
#     combination per rolling step.
#
# Action teacher runs on the FULL live window (teacher_mode=all) and
# aux losses follow the DMD random-slot pick (aux_loss_slot_policy=
# match_dmd) so the critic/probe supervise the same slot the DMD
# signal hits.
#
#   * num_live_slots         = 2
#   * passes_per_step        = 2
#   * denoising_step_list    = [500, 1000]
#   * action_decay_per_slot  = [1.0, 0.5]
#   * real_score_num_gt_chunks = 4  (slot 0: 4 GT + kv_anchor_chunk +
#       live_0 target. Slot 1: 4 GT + prev_0_student_detached + live_1.)
#   * dmd_active_slot_policy = random_any, dmd_num_active_slots = 1
#   * action_teacher_mode    = all
#   * aux_loss_slot_policy   = match_dmd
#
# Memory: 2-slot live window = 6 frames (~half of 4-slot's 12 frames)
# but we run 2 forwards per step. One of the two is under no_grad so
# activations aren't saved → expected peak memory ≈ 1-slot-forward + 1
# no-grad-2-slot-forward, so below the 4-slot baseline.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/longlive_phase1_rolling_staircase.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/phase1_5h_2x2_random
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
echo "Starting phase-1 2x2 random-slot/pass DMD (1 node, 4 GPUs, 5h) on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# NOTE: slots_per_rank here is the independent-ride batcher's slots
# (one ride per ride-slot on a rank). It is orthogonal to
# num_live_slots (which is the rolling-window width within a ride).
# We keep it at 4 so each rank carries 4 independent rides as before.
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
    log_interval=2 \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_weu \
    caption_root=/projects/u6ex/fbots/frodobots_captions/train \
    motion_root=/projects/u6ex/fbots/frodobots_motion \
    min_ride_frames=80 \
    slots_per_rank=4 \
    rolling_steps_per_iter=16 \
    num_live_slots=2 \
    passes_per_step=2 \
    "denoising_step_list=[500,1000]" \
    "action_decay_per_slot=[1.0,0.5]" \
    real_score_num_gt_chunks=4 \
    dmd_active_slot_policy=random_any \
    dmd_num_active_slots=1 \
    action_teacher_mode=all \
    aux_loss_slot_policy=match_dmd

echo "Training completed on $(date)"
