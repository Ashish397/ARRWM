#!/bin/bash
#SBATCH --job-name=z-critic-v11-cross-attn-probes-32gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.err

# =====================================================================
# v11 — cross-attention state probes + aggressive action weights
#
# Architecture change from v10:
#   - REPLACED per-frame state tokens with cross-attention probes
#   - 6 probe layers tap transformer at evenly-spaced depths
#   - Learned queries (one per chunk) progressively refine via
#     cross-attention to noisy-side visual features at each tap
#   - probe_dim=256, 8 heads, ~9M lightweight params
#   - No state tokens in the main sequence → shorter sequence,
#     no interference with denoising self-attention
#
# Weight changes from v10 (same as v11-aggressive):
#   - action_critic_z_loss_weight:          0.2  -> 1.0   (5x)
#   - generator_action_z_guidance_weight:   0.15 -> 1.5   (10x)
#   - state_head_loss_weight:               0.05 -> 1.5   (30x)
#   - state_guidance_weight:                0.2  -> 0.5   (2.5x)
#   - state_head_out_dim:                   8    -> 2     (z2/z7 only)
#   - z_guidance_warmup_steps:              500  -> 200   (shorter ramp)
#   - state_guidance_warmup_steps:          500  -> 200   (shorter ramp)
#
# Unchanged from v10:
#   - 8 nodes x 4 GPUs = 32 GPUs
#   - LoRA rank 256, alpha 256
#   - Generator lr 5e-5, critic lr 3e-4
#   - critic_updates_per_step = 2
#   - streaming_chunk_size 21, context_frames 3
#   - 3D CNN ActionCritic: base_channels 128, 4 res blocks
#   - Independent per-slot batcher, staggered start offsets
# =====================================================================

set -e -o pipefail

cd /scratch/u6ej/as1748.u6ej/ARRWM

CONFIG=configs/causal_lora_diffusion_teacher.yaml
LOGDIR=/scratch/u6ej/as1748.u6ej/ARRWM/logs/z_critic_v11_cross_attn_probes
WANDB_SAVE_DIR=wandb

source /scratch/u6ej/as1748.u6ej/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ej/as1748.u6ej/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

mkdir -p "$LOGDIR"
mkdir -p logs

# Reuse ride manifest from v10 to skip the ~20-minute zarr scan
V10_MANIFEST=/scratch/u6ej/as1748.u6ej/ARRWM/logs/z_critic_v10_state_tokens/.ride_manifest.pt
if [ -f "$V10_MANIFEST" ] && [ ! -e "$LOGDIR/.ride_manifest.pt" ]; then
  ln -s "$V10_MANIFEST" "$LOGDIR/.ride_manifest.pt"
  echo "Symlinked ride manifest cache from v10"
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CONFIG=$CONFIG"
echo "Starting z-critic v11 (cross-attention probes) training on $(date)"
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
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --no-auto-resume

echo "Training completed on $(date)"
