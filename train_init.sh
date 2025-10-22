#!/bin/bash
#SBATCH --job-name=longlive-init
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --time=8:19:00
#SBATCH --output=logs/%x_%j.out

# Change to the ARRWM directory where train.py is located
cd /home/u5as/as1748.u5as/frodobots/ARRWM

# Activate arrwm conda environment
if [ -f "$HOME/miniforge/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge/etc/profile.d/conda.sh"
else
  echo 'conda.sh not found at $HOME/miniforge/etc/profile.d/conda.sh' >&2
  exit 1
fi
conda activate arrwm

# Project path and config
CONFIG=configs/longlive_train_init_real.yaml
BASE_LOGDIR=/scratch/u5as/as1748.u5as/frodobots/dmd2/logs
EXPERIMENT_NAME=$(python - <<'PY_BLOCK'
from omegaconf import OmegaConf
config = OmegaConf.load('configs/longlive_train_init_real.yaml')
print(config.get('config_name', 'dmd2-experiment'))
PY_BLOCK
)
RUN_ID=$(date +%Y%m%d-%H%M%S)
LOGDIR="$BASE_LOGDIR/${EXPERIMENT_NAME}_${RUN_ID}"
WANDB_SAVE_DIR=wandb

mkdir -p "$BASE_LOGDIR"
mkdir -p "$LOGDIR"

if [ -n "$SLURM_JOB_NODELIST" ]; then
  MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
  NODE_RANK=${SLURM_NODEID}
  NNODES=${SLURM_JOB_NUM_NODES}
  NPROC_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
else
  MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
  NODE_RANK=${NODE_RANK:-0}
  NNODES=${NNODES:-1}
  NPROC_PER_NODE=${NPROC_PER_NODE:-4}
fi
MASTER_PORT=${MASTER_PORT:-29500}

export CONFIG LOGDIR WANDB_SAVE_DIR EXPERIMENT_NAME MASTER_ADDR MASTER_PORT NNODES NPROC_PER_NODE

# Ensure Hugging Face downloads use scratch space (larger quota than $HOME).
CACHE_DIR='/scratch/u5as/as1748.u5as/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

echo "CONFIG=$CONFIG"

srun --ntasks=$NNODES --ntasks-per-node=1 \
  bash -c '
    torchrun \
      --nproc_per_node=${NPROC_PER_NODE} \
      --nnodes=${NNODES} \
      --node_rank=${SLURM_NODEID} \
      --master_addr=${MASTER_ADDR} \
      --master_port=${MASTER_PORT} \
      train.py \
        --config_path ${CONFIG} \
        --logdir ${LOGDIR} \
        --wandb-save-dir ${WANDB_SAVE_DIR} \
        --experiment-name ${EXPERIMENT_NAME}'

