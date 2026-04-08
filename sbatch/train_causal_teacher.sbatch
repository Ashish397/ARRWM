#!/bin/bash
#SBATCH --job-name=causal-teacher-4gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.err

set -e -o pipefail

cd /scratch/u6ej/as1748.u6ej/ARRWM

CONFIG=configs/causal_lora_diffusion_teacher.yaml
LOGDIR=/scratch/u6ej/as1748.u6ej/ARRWM/logs/causal_lora_teacher
WANDB_SAVE_DIR=wandb
EXPERIMENT_NAME="causal-teacher-$(date +%Y%m%d-%H%M%S)"

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

echo "CONFIG=$CONFIG"
echo "Starting causal teacher training on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $(hostname)"
echo "GPUs: $SLURM_GPUS_ON_NODE"

torchrun \
  --nproc_per_node=4 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --experiment-name "$EXPERIMENT_NAME"

echo "Training completed on $(date)"
