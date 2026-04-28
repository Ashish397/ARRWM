#!/bin/bash
# Launcher for the streaming-vs-diag pred_real comparison harness.
set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

set --
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH

PORT=$((30000 + RANDOM % 20000))
torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --master_addr=127.0.0.1 \
  --master_port=$PORT \
  _diag_p1/compare_streaming_vs_diag.py
