#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-freeze-16h-8node
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing DMD — DEFAULTS + TEACHER-FREEZE MASK,
# 1 node x 4 GPUs, 16h.
#
# Same as ``train_action_forcing_phase1_16h.sbatch`` (pure DMD, no
# aux losses, no R3GAN, no SC-DMD) PLUS per-frame teacher-freeze
# rejection on the DMD gradient.
#
# Why this exists. The v14 ``real_score`` (frozen LoRA-merged
# bidirectional teacher) occasionally outputs a single hazy / frozen
# latent frame inside the 21-frame DMD scoring window. Visible as a
# washed-out frame in ``_diag_p1/out/F_real_GT.mp4`` around the
# 3-second mark. If we let DMD score against that frame, the
# (fake - real) gradient pulls the student toward the bad teacher
# pred and the student learns to reproduce the freeze.
#
# Detection. Inside ``compute_distribution_matching_loss`` we now
# compute per-frame MAE = mean(|pred_real_image - gt_target|) where
# ``gt_target`` is the GT video at the same time positions as the
# noisy_x window. Per batch we take the median across frames as the
# robust reference (the freeze frame itself is the outlier so the
# median is unaffected). Frames whose MAE exceeds
# ``threshold * median(MAE)`` are flagged and ANDed into the
# existing ``gradient_mask`` so the DMD MSE skips them entirely.
#
# Knobs (configs/action_forcing_phase1_freeze.yaml):
#   * teacher_freeze_detect_enabled: true       (opt-in flag)
#   * teacher_freeze_threshold: 2.0             (multiplier on median;
#                                                must be > 1.0)
#
# Per-iter wandb metrics (gen/ namespace):
#   gen/teacher_freeze_count       — int, frames flagged this iter
#   gen/teacher_freeze_rate        — fraction of B*F frames flagged
#   gen/teacher_freeze_max_mae     — max per-frame MAE
#   gen/teacher_freeze_median_mae  — median per-frame MAE
#
# Sister sbatches:
#   sbatch/train_action_forcing_phase1_16h.sbatch        (defaults only)
#   sbatch/train_action_forcing_phase1_gan_16h.sbatch    (defaults + R3GAN)
#   sbatch/train_action_forcing_phase1_scdmd_16h.sbatch  (defaults + SC-DMD)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_freeze.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_freeze_16h
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
echo "Starting Phase-1 Action-Forcing DMD + teacher-freeze mask (1 node x 4 GPUs, 16h) on $(date)"

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
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$LOGDIR \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=action_forcing_phase1_freeze_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
