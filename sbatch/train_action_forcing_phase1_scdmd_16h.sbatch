#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-scdmd-16h-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing DMD — DEFAULTS + SC-DMD, 1 node x 4 GPUs, 16h.
#
# Same as ``train_action_forcing_phase1_16h.sbatch`` (pure DMD, no
# aux losses, no R3GAN) PLUS the Salt SC-DMD semigroup defect
# regularizer:
#
#     L_SC = E[ || Ψ_θ^{ts→te}(x_ts) - Ψ_θ^{tm→te}(Ψ_θ^{ts→tm}(x_ts)) ||² ]
#
# enforcing local consistency of the student's velocity field across
# composed denoising paths. Recipe ported verbatim from
# action_forcing_phase1_aux_scdmd.yaml, minus the action critic /
# motion teacher (this config is "defaults + SC-DMD only", clean A/B
# vs the defaults sbatch).
#
# SC-DMD knobs (configs/action_forcing_phase1_scdmd.yaml):
#   * sc_dmd_loss_weight: 0.05      (~5% of DMD's typical magnitude)
#   * sc_dmd_warmup_steps: 500      (linear ramp 0 → sc_dmd_loss_weight)
#
# Implementation: ``ActionForcingDMD.sc_dmd_loss`` — one chunk-0
# forward with a fresh ``num_frame_per_block``-frame KV cache, two
# extra DiT forwards on a 3-frame chunk per gen step. Cost: ~1-2%
# extra wallclock end-to-end vs the defaults sbatch.
#
# Sister sbatches:
#   sbatch/train_action_forcing_phase1_16h.sbatch      (defaults only)
#   sbatch/train_action_forcing_phase1_gan_16h.sbatch  (defaults + R3GAN)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_scdmd.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_scdmd_16h_v2
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
echo "Starting Phase-1 Action-Forcing DMD + SC-DMD (1 node x 4 GPUs, 16h) on $(date)"

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
    run_name=action_forcing_phase1_scdmd_j${SLURM_JOB_ID}_v2

echo "Training completed on $(date)"
