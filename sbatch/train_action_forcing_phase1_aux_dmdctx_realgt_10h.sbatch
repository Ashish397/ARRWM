#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-aux-dmdctx-realgt-10h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing DMD WITH AUX + DMD_CONTEXT (A1) + DMD_REAL_GT.
# 1 node x 4 GPUs, 10h.
#
# Same recipe as ``sbatch/train_action_forcing_phase1_aux_dmdctx_10h.sbatch``
# (Option A1 v14 teacher-forcing parity: cf=3 clean GT context frames +
# 21 noisy student frames per scorer forward, context_shift=1) PLUS:
#
#   * ``dmd_real_GT=true``: real_score's clean_x is replaced with a
#     LIGHTLY NOISED GT view (same shape, same memory, same KV cost).
#     fake_score keeps PURE clean GT (aug_t=0) for both DMD's
#     ``_compute_kl_grad`` AND for ``critic_loss``'s independent
#     fake-score training step.
#   * ``dmd_real_GT_aug_t=20``: noise level in scheduler timesteps
#     (out of 1000). At t=20 the flow scheduler's sigma is ~0.02
#     (~2% noise + ~98% GT) — essentially imperceptible. The init
#     log prints the effective sigma to confirm at runtime.
#
# Cost vs the AUX + dmd_context (no real_GT) baseline:
#   * Rollout compute:  UNCHANGED (still 21 student frames).
#   * Scorer FLOPs:     UNCHANGED (clean_x is still 21 frames).
#   * Memory:           +1 tensor of clean_x's size (the noised view).
#                       Negligible (~MBs vs the model's GBs).
#   * Wallclock:        +1 ``scheduler.add_noise`` per generator step.
#                       <1% extra wallclock vs dmd_context.
#
# Wandb metrics (same as AUX + dmd_context baseline; no new metrics
# specifically for dmd_real_GT — the noise level is a constant set by
# config, the generator/critic loss curves should drift slightly vs
# the symmetric dmd_context A/B).
#
# To run the dmd_context-only A/B (no dmd_real_GT, both scorers see
# pure clean GT):
#   sbatch sbatch/train_action_forcing_phase1_aux_dmdctx_10h.sbatch
# To run the AUX-only A/B (no dmd_context, no dmd_real_GT):
#   sbatch sbatch/train_action_forcing_phase1_aux_10h.sbatch
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_aux_dmdctx_realgt.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_aux_dmdctx_realgt_10h
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
echo "Starting phase-1 Action-Forcing DMD + AUX + DMD_CONTEXT + DMD_REAL_GT (1 node, 4 GPUs, 10h) on $(date)"

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
    run_name=action_forcing_phase1_aux_dmdctx_realgt_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
