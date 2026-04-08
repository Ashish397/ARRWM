#!/bin/bash
#SBATCH --job-name=reward-critic-v3-16gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.err

# =====================================================================
# v3 changes from v2 (causal-teacher-v2-16gpu / run 66ioc7i5):
#
# Critic overhaul – "reward critic":
#   - Dropped z_head (action latent regression) entirely
#   - Critic now predicts teacher reward only (reward_head)
#   - Critic receives diffusion timestep t (sinusoidal embed)
#   - Critic receives commanded action z2/z7 (action embed)
#
# Generator reward guidance – NEW, was disabled in v2:
#   - Differentiable reward gradients flow from critic to generator
#   - generator_action_reward_guidance_weight = 1.0
#   - 500-step linear warmup after LR warmup
#
# Asymmetric reward:
#   - Overshoot (same dir, bigger mag) weighted 0.5
#   - Undershoot weighted 1.0
#   - Near-zero targets fall back to symmetric |err|
#
# Unchanged from v2:
#   - 4 nodes x 4 GPUs = 16 GPUs
#   - LoRA rank 256, alpha 256
#   - Generator lr 5e-5, critic lr 3e-4
#   - streaming_chunk_size 21, context_frames 3
#   - Critic arch: base_channels 128, 4 res blocks, SE, AttentionPool
# =====================================================================

set -e -o pipefail

cd /scratch/u6ej/as1748.u6ej/ARRWM

CONFIG=configs/causal_lora_diffusion_teacher.yaml
LOGDIR=/scratch/u6ej/as1748.u6ej/ARRWM/logs/reward_critic_v3
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

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CONFIG=$CONFIG"
echo "Starting reward-critic v3 training on $(date)"
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
