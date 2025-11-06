#!/bin/bash

# Route Hugging Face cache to scratch to avoid $HOME quota issues.
CACHE_DIR='/scratch/u5dk/as1748.u5dk/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

torchrun \
  --nproc_per_node=1 \
  --master_port=29500 \
  inference.py \
  --config_path configs/longlive_inference.yaml