#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-freeze-gan-scdmd-16h-8node
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
# Phase-1 Action-Forcing DMD — TEACHER-FREEZE MASK + R3GAN + SC-DMD,
# 8 nodes x 4 GPUs, 16h.
#
# Combines all three opt-in losses on top of the freeze base:
#   * Teacher-freeze mask  — from configs/action_forcing_phase1_freeze.yaml
#                            (rejects per-frame DMD signal where the
#                            frozen teacher emits a bad/hazy frame).
#   * R3GAN                — RpGAN + R1 + R2 zero-centered grad penalty
#                            on GT vs student latent video.
#   * SC-DMD               — Salt-paper semigroup defect regularizer
#                            ``L_SC = E[||Ψ^{ts→te} − Ψ^{tm→te}∘Ψ^{ts→tm}||²]``
#                            penalizing DMD's compositionality gap.
#
# Why this exists. The four sister sbatches (defaults / freeze / gan /
# scdmd) are clean A/Bs that each toggle exactly one knob over the
# defaults. This one is the "all on" stack — useful as a maxed-out
# baseline to compare against the singletons, and as the realistic
# training recipe if all three losses turn out to help additively.
#
# Knobs are CLI-overridden on top of the freeze YAML so this sbatch
# stays in lockstep with the freeze base (max_rolls_per_ride=60,
# force_exit_step_enabled=false, K=1-per-step state machine, async
# CPU prefetch, etc.) without forking another YAML.
#
# R3GAN knobs (from action_forcing_phase1_gan.yaml):
#   * gan_loss_weight: 0.05         (G-side RpGAN, ~5% of DMD magnitude)
#   * gan_r1_gamma: 1.0
#   * gan_r2_gamma: 1.0             (R3GAN paper requires both R1 + R2)
#   * gan_warmup_steps: 500         (linear ramp on G; D updates from step 0)
#   * gan_updates_per_step: 1       (1:1 D:G ratio)
#   * gan_lr: 2e-4, betas=[0.0, 0.9] (StyleGAN/R3GAN convention)
#
# SC-DMD knobs (from action_forcing_phase1_scdmd.yaml):
#   * sc_dmd_loss_weight: 0.05      (~5% of DMD magnitude at steady state)
#   * sc_dmd_warmup_steps: 500      (linear ramp 0 → loss_weight)
#
# Memory cost over freeze sbatch: ~1-2 GB (R3GAN disc + R1/R2 second-
# order graph) + ~minimal (SC-DMD does 2 extra DiT forwards on a 3-
# frame chunk). Wallclock cost: ~50-80 ms/iter for R3GAN's D update +
# frozen-D forward; ~1-2% for SC-DMD (single chunk, fresh KV cache).
#
# Sister sbatches:
#   sbatch/train_action_forcing_phase1_16h_8node.sbatch        (defaults only)
#   sbatch/train_action_forcing_phase1_freeze_16h_8node.sbatch (defaults + teacher-freeze mask)
#   sbatch/train_action_forcing_phase1_gan_16h_8node.sbatch    (defaults + R3GAN)
#   sbatch/train_action_forcing_phase1_scdmd_16h_8node.sbatch  (defaults + SC-DMD)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_freeze.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_freeze_gan_scdmd_16h
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
echo "Starting Phase-1 Action-Forcing DMD + freeze + R3GAN + SC-DMD (8 nodes x 4 GPUs, 16h) on $(date)"

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
    real_guidance_scale=0.0 \
    action_teacher_mode=all \
    max_rolls_per_ride=300 \
    dfake_gen_update_ratio=1 \
    streaming_max_length=1500 \
    max_ride_frames=1490 \
    run_name=action_forcing_phase1_freeze_gan_scdmd_j${SLURM_JOB_ID} \
    gan_enabled=true \
    gan_disc_in_channels=16 \
    gan_disc_base_channels=64 \
    gan_disc_num_blocks=4 \
    gan_loss_weight=0.05 \
    gan_r1_gamma=1.0 \
    gan_r2_gamma=1.0 \
    gan_warmup_steps=500 \
    gan_updates_per_step=1 \
    gan_lr=2.0e-04 \
    gan_betas='[0.0, 0.9]' \
    gan_eps=1.0e-08 \
    gan_weight_decay=0.0 \
    gan_max_grad_norm=10.0 \
    sc_dmd_enabled=true \
    sc_dmd_loss_weight=0.05 \
    sc_dmd_warmup_steps=500

echo "Training completed on $(date)"
