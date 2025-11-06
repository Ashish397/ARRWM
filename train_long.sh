#!/bin/bash

# Project path and config
CONFIG=configs/longlive_train_long.yaml
LOGDIR=/scratch/u5dk/as1748.u5dk/frodobots/dmd2/logs
WANDB_SAVE_DIR=wandb
EXPERIMENT_NAME="dmd2-long-$(date +%Y%m%d-%H%M%S)"

# Ensure Hugging Face downloads use scratch space (larger quota than $HOME).
CACHE_DIR='/scratch/u5dk/as1748.u5dk/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

echo "CONFIG=$CONFIG"

torchrun \
  --nproc_per_node=8 \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --experiment-name "$EXPERIMENT_NAME"