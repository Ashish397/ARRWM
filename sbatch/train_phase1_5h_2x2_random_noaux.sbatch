#!/bin/bash
#SBATCH --job-name=phase1-5h-2x2-random-noaux
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
# VARIANT: 2 slots × 2 passes/step, NO-AUX ablation.
# Identical to the 2x2 random sbatch except every action-supervision
# head is stripped:
#
#   DISABLED via --override:
#     * action_critic_aux_enabled = false
#         → no critic instantiation, no critic aux loss.
#     * state_probe_aux_enabled   = false
#         → no state-probe module attached to the generator wrapper;
#           the probe forward inside wan_wrapper is bypassed.
#     * action_teacher_mode       = off
#         → no CoTracker + VAE forward; no ss_vae encoder; no teacher-z
#           targets. The DMD model's auto-off-switch will also force
#           both aux heads to False at build time even if the two
#           *_aux_enabled flags were left on — we belt-and-braces them.
#
#   KEPT ON AND TRAINABLE:
#     * action_projection (Stream A → AdaLN modulation): on + trainable.
#     * action_token_projection (Stream B → per-frame action tokens):
#         on + trainable.
#     * DiT generator + fake_score + real_score.
#
# Purpose: isolate what the 2x2 staircase learns from pure DMD + the
# two trainable action-projection MLPs, with NO motion-pipeline
# supervision hint. Paired with train_phase1_5h_2x2_random.sbatch for
# direct aux-on-vs-off comparison on identical ladder geometry.
#
# Same rolling-staircase topology as the 2x2-random main sbatch:
#   * num_live_slots         = 2
#   * passes_per_step        = 2
#   * denoising_step_list    = [500, 1000]
#   * action_decay_per_slot  = [1.0, 0.5]
#   * real_score_num_gt_chunks = 4
#   * dmd_active_slot_policy = random_any, dmd_num_active_slots = 1
#     (truly-uniform single-slot pick per step, DDP-synced)
#
# Dataset: WEU zarrs, min_ride_frames=80 (explicit overrides, not
# YAML-dependent).
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/longlive_phase1_rolling_staircase.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/phase1_5h_2x2_random_noaux
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

# No CoTracker loaded (action_teacher_mode=off), so no ~/.cache/torch
# access is needed. TORCH_HOME left untouched.

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
echo "Starting phase-1 2x2 random-slot/pass DMD NO-AUX (1 node, 4 GPUs, 5h) on $(date)"

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
    num_live_slots=2 \
    passes_per_step=2 \
    "denoising_step_list=[500,1000]" \
    "action_decay_per_slot=[1.0,0.5]" \
    real_score_num_gt_chunks=4 \
    dmd_active_slot_policy=random_any \
    dmd_num_active_slots=1 \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    train_action_projection=true \
    train_action_token_projection=true

echo "Training completed on $(date)"
