#!/bin/bash
#SBATCH --job-name=v14e-pca4
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# v14e: VAE OFF -> RAW top-8 PCA actions (pca_0=throttle, pca_1=steer).
# Critic supervises top-8 PCA. Guidance decoupled from LR warmup (LR warmup
# 200; guidance start 50, ramp 50). Eval = GT-vs-flip fast causal chain.
# 8 nodes = global batch 32. 4h via --time; ckpt at step 1 + every 100.
# =====================================================================

set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/causal_lora_diffusion_teacher_v14e_pca4.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca4
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
export ARRWM_ACTION_ENCODER=pca_raw   # dataset conditioning = raw PCA (must match teacher_action_encoder)

echo "CONFIG=$CONFIG"
echo "Starting v14e-pca4 (VAE off, raw PCA) on $(date); ARRWM_ACTION_ENCODER=$ARRWM_ACTION_ENCODER"
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
