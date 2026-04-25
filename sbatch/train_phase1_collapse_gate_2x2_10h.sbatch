#!/bin/bash
#SBATCH --job-name=phase1-collapse-2x2-10h
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
# Phase-1 Rolling-Staircase DMD, 1 node x 4 GPUs, 10h.
#
# Same as train_phase1_collapse_gate_10h.sbatch, but 2x2 live geometry:
#   num_live_slots=2, passes_per_step=2  (NS*P=4, same ODE budget as 4x1).
#   action_decay_per_slot=[1.0, 0.5]    (length must match num_live_slots).
# Base YAML is unchanged: denoising_step_list [1000, 625, 500, 312.5] is
# already length 4 = NSP; collapse gate, curriculum, and other knobs match
# configs/longlive_phase1_rolling_staircase.yaml.
#
# This sbatch sets paths (log_dir, wandb_dir), wandb run_name (distinct
# from train_phase1_collapse_gate_10h.sbatch), plus the slot overrides
# above; everything else stays in the YAML.
#
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/longlive_phase1_rolling_staircase.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/phase1_collapse_gate_10h_2x2
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
echo "Geometry: 2x2 (num_live_slots=2, passes_per_step=2)"
echo "Starting phase-1 collapse-gate + curriculum DMD 2x2 (1 node, 4 GPUs, 10h) on $(date)"

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
  trainer/causal_rolling_staircase_train.py \
  --config $CONFIG \
  --override \
    log_dir=$LOGDIR \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=phase1_collapse_2x2_j${SLURM_JOB_ID} \
    num_live_slots=2 \
    passes_per_step=2 \
    'action_decay_per_slot=[1.0,0.5]'

echo "Training completed on $(date)"
