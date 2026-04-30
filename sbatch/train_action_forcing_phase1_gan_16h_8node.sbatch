#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-gan-16h-8node
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
# Phase-1 Action-Forcing DMD — DEFAULTS + R3GAN, 1 node x 4 GPUs, 16h.
#
# Same as ``train_action_forcing_phase1_16h.sbatch`` (pure DMD, no
# aux losses, no SC-DMD) PLUS the R3GAN discriminator (RpGAN + R1 +
# R2 zero-centered grad penalty) trained on GT vs student latent
# video. Recipe ported verbatim from action_forcing_phase1_aux_gan.yaml,
# minus the action critic / motion teacher (this config is "defaults +
# R3GAN only", clean A/B vs the defaults sbatch).
#
# R3GAN knobs (configs/action_forcing_phase1_gan.yaml):
#   * gan_loss_weight: 0.05         (G-side RpGAN, ~5% of DMD magnitude)
#   * gan_r1_gamma: 1.0
#   * gan_r2_gamma: 1.0             (R3GAN paper requires both R1 + R2)
#   * gan_warmup_steps: 500         (linear ramp on the G side; D
#                                    updates from step 0 regardless)
#   * gan_updates_per_step: 1       (1:1 D:G ratio)
#   * gan_lr: 2e-4, betas=[0.0, 0.9] (StyleGAN/R3GAN convention)
#
# Memory cost over the defaults sbatch: ~1-2 GB for the discriminator
# parameters + activations + the R1/R2 second-order graph. Wallclock
# cost: 1 D forward + 1 D backward (with create_graph for R1/R2) + 1
# frozen-D forward for G's term — typically ~50-80 ms/iter at
# gan_disc_base_channels=64 on H100.
#
# Sister sbatches:
#   sbatch/train_action_forcing_phase1_16h.sbatch        (defaults only)
#   sbatch/train_action_forcing_phase1_scdmd_16h.sbatch  (defaults + SC-DMD)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_16h
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
echo "Starting Phase-1 Action-Forcing DMD + R3GAN (1 node x 4 GPUs, 16h) on $(date)"

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
    run_name=action_forcing_phase1_gan_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
