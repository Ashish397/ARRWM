#!/bin/bash
#SBATCH --job-name=phase1-5h-4x1-random
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
# VARIANT: 4 slots × 1 pass/step. Each rolling step applies DMD gradient
# to ONE truly-random slot (uniform draw from {0..3}, DDP-synced).
# Action teacher runs on slot 0 only (teacher_mode=slot0) so the
# critic/probe supervision is pinned to slot 0 regardless of the
# DMD random-slot pick.
#
#   * num_live_slots         = 4
#   * passes_per_step        = 1
#   * denoising_step_list    = [250,500,750,1000]
#   * action_decay_per_slot  = [1.0, 0.75, 0.5, 0.25]
#   * real_score_num_gt_chunks = 4  (slot 0: 4 GT + kv_anchor_chunk
#       [= most recently committed KV chunk] + live_0 target. Slots
#       1..3: 4 GT + prev_{n-1}_student_detached + live_n.)
#   * dmd_active_slot_policy = random_any, dmd_num_active_slots = 1
#       → truly uniform 1-slot pick per step, DDP-synced.
#   * action_teacher_mode    = slot0
#       → CoTracker + VAE run on the full live window for tracking
#         quality; critic/probe aux losses consume slot-0's teacher-z
#         only. Other slots receive zero aux loss this step.
#
# Dataset, KV geometry, viz cadence (30 min + step-0), checkpoint paths,
# ride sorting, and min-ride-frames=160 inherit from the YAML.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/longlive_phase1_rolling_staircase.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/phase1_5h_4x1_random
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

# Keep ~/.cache/torch (CoTracker3 offline weights + repo cache) on its
# default path — the YAML's cotracker_source_dir / cotracker_checkpoint_path
# point there so no network call fires.

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
echo "Starting phase-1 4x1 random-slot DMD (1 node, 4 GPUs, 5h) on $(date)"

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
    log_interval=2 \
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
    dmd_active_slot_policy=random_any \
    dmd_num_active_slots=1 \
    action_teacher_mode=slot0 \
    aux_loss_slot_policy=slot0

echo "Training completed on $(date)"
