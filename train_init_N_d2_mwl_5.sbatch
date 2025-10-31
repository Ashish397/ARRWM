#!/bin/bash
#SBATCH -J train_init_real_N_d2_mwl_5_with_actions
#SBATCH --nodes=16
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=4
#SBATCH -t 22:00:00
#SBATCH --requeue
#SBATCH -o %x_%j.out
#SBATCH -e %x_%j.err

set -e -o pipefail

# Change to the ARRWM directory where train.py is located
cd /home/u5as/as1748.u5as/frodobots/ARRWM

# Project path and config
CONFIG=configs/longlive_train_init_real_N_d2_mwl_5.yaml
LOGDIR=/scratch/u5as/as1748.u5as/frodobots/dmd2/logs_N_d2_mwl_5_with_actions
WANDB_SAVE_DIR=wandb
EXPERIMENT_NAME="dmd2-init-N-d2-mwl-5-with-actions-$(date +%Y%m%d-%H%M%S)"

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

echo "CONFIG=$CONFIG"
echo "Starting training job on $(date)"
echo "Job ID: $SLURM_JOB_ID"

torchrun \
  --nproc_per_node=4 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --experiment-name "$EXPERIMENT_NAME"

echo "Training completed on $(date)"
