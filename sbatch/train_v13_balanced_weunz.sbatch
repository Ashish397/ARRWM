#!/bin/bash
#SBATCH --job-name=v13-balanced-mrs-32gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# v13 — balanced loss weights + MRS subset + shuffled windows
#
# Changes from v12:
#   - Loss weights pulled back toward v10 levels to prioritise image quality
#     over aggressive action following:
#       action_critic_z_loss_weight:         1.0  -> 0.5
#       generator_action_z_guidance_weight:  1.5  -> 0.5
#       state_head_loss_weight:              1.5  -> 0.3
#       state_guidance_weight:               0.5  -> 0.3
#       state_guidance_action_scale:         1.1  -> 1.0 (remove overshoot)
#   - Dataset: MRS only (Madrid+Rome+Stockholm, 574 rides, ~140h)
#     Filter: claude/city_zarrs.csv, zarrs from frodobots_encoded/
#   - Batcher: shuffled windows across rides (not sequential per ride)
#   - Initialised from v12 checkpoint (step 3250)
#
# Training budget:
#   8 nodes × 4 GPUs = 32 GPUs, 24h
#   ~10s/step → 360 steps/hr → 11,520 windows/hr → 276,480 windows in 24h
#   MRS has ~140h → ~2.88 passes in 24h
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/causal_lora_diffusion_teacher_v13.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v13_balanced_weunz
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
echo "Starting v13 (balanced WEUNZ) training on $(date)"

srun --ntasks-per-node=1 bash -c 'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' || true
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
