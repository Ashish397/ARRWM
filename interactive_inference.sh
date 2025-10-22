#!/bin/bash

# Route Hugging Face cache to scratch to avoid $HOME quota issues.
CACHE_DIR='/scratch/u5as/as1748.u5as/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

torchrun \
  --nproc_per_node=1 \
  --master_port=29500 \
  interactive_inference.py \
  --config_path configs/longlive_interactive_inference.yaml