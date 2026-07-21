#!/bin/bash
#SBATCH --job-name=v14d-ctrl-8node
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:30:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# CONTROL TEST (8node): 14d recipe VERBATIM except global batch (= node
# count) and the sqrt-scaled PEAK lr. global batch 32, lr 4.95e-5 (sqrt). Runs to step 600 via stop_at_step (schedule untouched), ckpt at 250+600.
# Schedule shapes (warmup 200, cosine/30000) IDENTICAL to 14d; run cut off
# by THIS --time, not by config. Every 20 steps the first control_n_videos
# ranks each render one constant-8-direction causal window to
# <logdir>/control_test/. NO checkpoints (save_checkpoints:false).
# =====================================================================

set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/causal_lora_diffusion_teacher_v14d_control_8node.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14d_control_test_8node
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
echo "Starting v14d CONTROL-TEST 8node on $(date)"
echo "Job ID: $SLURM_JOB_ID  Nodes: $SLURM_NNODES  Total GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
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
  --wandb-save-dir $WANDB_SAVE_DIR

echo "Training completed on $(date)"
