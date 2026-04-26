#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-aux-dmdctx-10h
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
# Phase-1 Action-Forcing DMD WITH AUX LOSSES + DMD_CONTEXT (A1).
# 1 node x 4 GPUs, 10h.
#
# Same recipe as ``sbatch/train_action_forcing_phase1_aux_10h.sbatch``
# (action critic + frozen motion teacher z-guidance, 1:1 DMD2 fake_score
# update cadence, MAE-driven dynamic rollout extension) PLUS the
# DMD_CONTEXT mode (Option A1 — full v14 alignment):
#
#   * ``rollout_frames=21`` (UNCHANGED from AUX). Student still rolls
#     21 latent frames per iter (= 7 chunks of 3); the rollout window
#     is shifted to ride frames ``[3..23]``.
#   * ``dmd_context=true``: feed ``real_score`` / ``fake_score`` a
#     clean GT context block alongside the noisy student rollout —
#     mirroring v14's training:
#       - clean_x  = ride[0..20]   (21 clean GT frames; aug_t=0)
#       - noisy_x  = student rollout (21 noisy student frames)
#     plus a clean conditional dict built from the GT context's
#     actions. Scorers attend over both halves jointly; only the
#     noisy half contributes to the DMD MSE.
#   * ``dmd_context_clean_frames=3`` (= cf): number of leading GT
#     frames handed to the scorers. Sets
#     ``model.{real,fake}_score.context_shift = cf // npb = 1`` to
#     align the scorer RoPE positions to the v14 LoRA weights. v14
#     was trained with cf=3, num_frames=21, 18-frame overlap.
#
# Why DMD_CONTEXT (A1):
#   v14's bidirectional teacher was trained with a 24-frame window
#   (3 clean context frames + 21 noisy target frames, 18-frame
#   overlap). Distilling DMD by feeding only 21 noisy frames into
#   ``real_score`` is OOD for v14's positional embeddings and skips
#   the clean-context conditioning the teacher learned to attend to.
#   ``dmd_context=true`` restores in-distribution input to the
#   scorers WITHOUT growing the student's rollout (which would be
#   "unfair" — bigger compute budget than the AUX baseline). The
#   compute hit is purely on the scorer forwards.
#
# Per-iter wallclock cost vs AUX baseline:
#   * Rollout compute: UNCHANGED (still 21 student frames).
#   * Scorer FLOPs: ~+15% (24 vs 21 frames per scorer forward).
#   * KV cache memory: UNCHANGED (still 21-frame student cache;
#     scorers are bidirectional and don't use a KV cache).
#   * Overall ~+10-15% slower per iter than action_forcing_phase1_aux.
#   * Action critic + cotracker teacher cost is unchanged (still
#     only operates on the trailing 21-frame student rollout).
#
# Wandb metrics (same as AUX baseline; ``gen/`` namespace + ``train/``
# from the critic helper):
#     train/critic_z_loss          (per-step critic training loss)
#     train/critic_loss            (= weight * critic_z_loss)
#     train/critic_z2_mse          (per-dim teacher MSE, z2)
#     train/critic_z7_mse          (per-dim teacher MSE, z7)
#     train/gen_z_loss             (gen-side guidance MSE)
#     train/gen_action_loss        (= guidance_scale * gen_z_loss)
#     train/teacher_z2_mean        (teacher z2 mean)
#     train/teacher_z7_mean        (teacher z7 mean)
#     train/z_guidance_scale       (ramped guidance scale)
#     train/teacher_unavailable    (1.0 on iters where teacher fwd
#                                   failed and aux was skipped)
#     gen/baseline_last_chunk_mae  (MAE on baseline last 3 frames)
#     gen/last_chunk_mae           (MAE on whatever the last chunk
#                                   was — baseline or extension)
#     gen/mae_extension_count      (number of extension chunks fired)
#     dmdtrain_gradient_norm       (|DMD grad|.mean())
#     timestep                     (sampled DMD t)
#
# To run the AUX-only A/B (no dmd_context, rollout_frames=21):
#   sbatch sbatch/train_action_forcing_phase1_aux_10h.sbatch
#
# To run pure-DMD baseline (no aux, no dmd_context):
#   sbatch sbatch/train_action_forcing_phase1_10h.sbatch
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_aux_dmdctx.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_aux_dmdctx_10h
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
echo "Starting phase-1 Action-Forcing DMD + AUX + DMD_CONTEXT (1 node, 4 GPUs, 10h) on $(date)"

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
    run_name=action_forcing_phase1_aux_dmdctx_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
