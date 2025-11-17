#!/bin/bash
#SBATCH -J train_init_real_N_d2_lam_v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH -t 05:00:00
#SBATCH --requeue
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -e -o pipefail

# Change to the ARRWM directory where train.py is located
cd /home/u5dk/as1748.u5dk/ARRWM

# Project path and config
CONFIG=configs/longlive_train_init_real_N_d2_lam_1.yaml
LOGDIR_ROOT=/scratch/u5dk/as1748.u5dk/frodobots/dmd2/logs_N_d2_lam_v2
WANDB_SAVE_DIR=wandb
EXPERIMENT_NAME="dmd2-init-N-d2-lam-v2"
LOGDIR="$LOGDIR_ROOT/$EXPERIMENT_NAME"
mkdir -p "$LOGDIR"

# Activate arrwm conda environment
source /scratch/u5dk/as1748.u5dk/miniforge3/bin/activate
conda activate arrwm

# Ensure Hugging Face downloads use scratch space (larger quota than $HOME).
CACHE_DIR='/scratch/u5dk/as1748.u5dk/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

echo "CONFIG=$CONFIG"
echo "Starting training job on $(date)"
echo "Job ID: $SLURM_JOB_ID"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=1
NPROC_PER_NODE=4
CPUS_PER_NODE=64
RDZV_ID=${SLURM_JOB_ID:-$RANDOM}
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))
EXPECTED_WORLD_SIZE=4
if [[ $WORLD_SIZE -ne $EXPECTED_WORLD_SIZE ]]; then
  echo "Error: expected $EXPECTED_WORLD_SIZE GPUs (global batch size 4) but scheduler provided $WORLD_SIZE." >&2
  exit 1
fi
export MASTER_ADDR MASTER_PORT RDZV_ID WORLD_SIZE

srun --nodes=$NNODES \
     --ntasks=$NNODES \
     --ntasks-per-node=1 \
     --cpus-per-task=$CPUS_PER_NODE \
     --gpus-per-task=$NPROC_PER_NODE \
  bash -c "torchrun \
    --nnodes=$NNODES \
    --node_rank=\$SLURM_NODEID \
    --nproc_per_node=$NPROC_PER_NODE \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --rdzv_id=$RDZV_ID \
    train.py \
    --config_path $CONFIG \
    --logdir $LOGDIR \
    --wandb-save-dir $WANDB_SAVE_DIR \
    --experiment-name \"$EXPERIMENT_NAME\""

echo "Training completed on $(date)"
