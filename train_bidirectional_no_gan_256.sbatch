#!/bin/bash
#SBATCH --job-name=wan-bidir-lora-nogan-256
#SBATCH --nodes=16
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:59
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -e -o pipefail

# Change to the ARRWM directory where the training scripts live
cd /home/u5as/as1748.u5as/frodobots/ARRWM

# Project path and config
CONFIG=configs/bidirectional_lora_no_gan_256.yaml
LOGDIR=/scratch/u5as/as1748.u5as/frodobots/retrain_no_gan_256/logs
WANDB_SAVE_DIR=wandb
EXPERIMENT_NAME="retrain-no-gan-$(date +%Y%m%d-%H%M%S)"


# Activate arrwm conda environment
if [ -f "$HOME/miniforge/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniforge/etc/profile.d/conda.sh"
else
  echo 'conda.sh not found at $HOME/miniforge/etc/profile.d/conda.sh' >&2
  exit 1
fi
conda activate arrwm

# Ensure Hugging Face downloads use scratch space (larger quota than $HOME).
CACHE_DIR='/scratch/u5as/as1748.u5as/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

mkdir -p "$LOGDIR"

echo "CONFIG=$CONFIG"
echo "Starting training job on $(date)"
echo "Job ID: $SLURM_JOB_ID"

torchrun \
  --nproc_per_node=4 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --experiment-name "$EXPERIMENT_NAME" \
