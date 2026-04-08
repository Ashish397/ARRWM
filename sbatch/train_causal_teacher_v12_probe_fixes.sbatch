#!/bin/bash
#SBATCH --job-name=z-critic-v12-probe-fixes-32gpu
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.err

# =====================================================================
# v12 — probe fixes on top of v11 cross-attention probes
#
# Fixes from v11:
#   1. bf16 readout: explicit F.linear with float32 cast on both
#      input and weights, preventing dtype mismatch
#   2. Action-token leakage: probe KV now strips action tokens,
#      attending only to spatial features per frame
#   3. Tap-count assertion: forward() asserts len(tapped_features)
#      matches len(probe_layers), catching silent zip truncation
#
# Otherwise identical to v11 (same weights, architecture, config).
# =====================================================================

set -e -o pipefail

cd /scratch/u6ej/as1748.u6ej/ARRWM

CONFIG=configs/causal_lora_diffusion_teacher.yaml
LOGDIR=/scratch/u6ej/as1748.u6ej/ARRWM/logs/z_critic_v12_probe_fixes
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

# Reuse ride manifest from v10 — live symlink, no copy
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
echo "Starting z-critic v12 (probe fixes) training on $(date)"

# GPU memory diagnostic: check baseline memory state before training
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
