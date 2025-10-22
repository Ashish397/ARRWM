#!/bin/bash
#SBATCH --job-name=longlive-test_reference
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --time=23:59:00
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
CONFIG=configs/longlive_train_init.yaml
LOGDIR=logs
WANDB_SAVE_DIR=wandb
echo "CONFIG="$CONFIG

torchrun \
  --nproc_per_node=4 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --disable-wandb