#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-aux-gan-10h
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
# Phase-1 Action-Forcing DMD WITH AUX LOSSES + R3GAN.
# 1 node x 4 GPUs, 10h.
#
# Same recipe as ``train_action_forcing_phase1_aux_10h.sbatch`` (CF
# chunkwise DMD + 21-frame action-forcing rollout + infinity-RoPE +
# MAE extensions + action critic + frozen motion teacher z-guidance)
# PLUS an R3GAN discriminator — RpGAN + R1 + R2 (Huang et al.,
# NeurIPS 2024).
#
# Real samples for D = ride["latents"][:, gen_window] (GT, detached).
# Fake samples for D = aux["pred_image"] (student rollout, detached).
# G's RpGAN-G term flows back into the generator through the
# ``pred_image -> generator`` graph; D's params are frozen for that
# branch (``requires_grad_(False)``) so DDP doesn't fire on D
# during the generator backward.
#
# Per generator step:
#   1. Roll the student through 21 frames + (optional) MAE extensions.
#   2. CF-parity DMD loss against v14 LoRA-merged real_score and the
#      generator-mirrored fake_score.
#   3. Run cotracker → ss_vae → teacher z (chunkwise [B,7,8]).
#   4. Train action critic (2 indep AdamW steps, lr=3e-4, weight=0.25).
#   5. Frozen-critic z-guidance (weight=0.125 ramped over 500 steps).
#   6. R3GAN D-update (RpGAN-D + R1 + R2 grad penalties; backward
#      into D only — fake samples detached).
#   7. R3GAN G-update term (frozen D forward on pred_image, RpGAN-G
#      added to gen loss; ramped over 500 steps via gan_warmup_steps).
#   8. Generator backward + EMA; fake_score backward.
#
# Wandb metrics added vs aux config (under ``gen/`` namespace):
#   gen/train/r3gan_d_loss           D's RpGAN main term
#   gen/train/r3gan_r1               R1 zero-centered grad penalty (real)
#   gen/train/r3gan_r2               R2 zero-centered grad penalty (fake)
#   gen/train/r3gan_d_real           D's mean logit on REAL (GT)
#   gen/train/r3gan_d_fake_detached  D's mean logit on FAKE (detached)
#   gen/train/r3gan_d_fake_for_g     D's mean logit on FAKE (G's path)
#   gen/train/r3gan_g_loss_raw       G's RpGAN-G main term (unweighted)
#   gen/train/r3gan_g_loss_weighted  what's actually added to gen loss
#   gen/train/r3gan_g_weight         current ramped G-side weight
#
# Health checks:
#   * ``r3gan_d_real`` should drift POSITIVE; ``r3gan_d_fake_*`` NEG.
#   * ``r3gan_r1 + r3gan_r2`` should be O(0.01-0.5) of ``r3gan_d_loss``
#     once trained; if dominant, drop ``gan_r1_gamma`` / ``gan_r2_gamma``.
#   * ``r3gan_d_loss`` should sit around log(2) ~= 0.69 at convergence
#     (RpGAN's Nash equilibrium). If it crashes to 0 → D dominates →
#     bump ``gan_loss_weight`` or drop ``gan_lr``. If it's stuck
#     well above log(2) → D is losing → opposite tweak.
#
# Memory cost vs aux config: ~1-2 GB for D + R1/R2 second-order
# graph. Per-iter wallclock: +50-100ms for D's forward/backward
# + R1/R2 + G's frozen-D forward.
#
# To run the AUX-only A/B (no GAN) instead:
#   sbatch sbatch/train_action_forcing_phase1_aux_10h.sbatch
# To run the pure-DMD A/B (no aux, no GAN) instead:
#   sbatch sbatch/train_action_forcing_phase1_10h.sbatch
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_aux_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_aux_gan_10h
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
echo "Starting phase-1 Action-Forcing DMD + AUX + R3GAN (1 node, 4 GPUs, 10h) on $(date)"

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
    run_name=action_forcing_phase1_aux_gan_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
