#!/bin/bash
# 20-step smoke on high-motion data, 2-GPU DDP. Mirrors the production
# 4-config sbatches (CFG=0 after the v14-uncond-OOD fix) — the goal is
# to see whether the post-fix student trajectory stays clean across
# 20 iters on motion-rich data.
set -e -o pipefail

LOG_DIR=${1:?log_dir}

cd /scratch/u6ex/as1748.u6ex/ARRWM
set --
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR

export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PORT=$((30000 + RANDOM % 20000))

# 2-GPU DDP. ``nproc_per_node=2`` matches the existing srun's GPU
# allocation. sample_interval=2 in the smoke yaml gives videos at
# steps 2, 4, ..., 20.
torchrun \
  --nnodes=1 \
  --nproc_per_node=2 \
  --master_addr=127.0.0.1 \
  --master_port=$PORT \
  trainer/causal_action_forcing_train.py \
  --config configs/smoke_extender.yaml \
  --override \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_smoke_high \
    log_dir=$LOG_DIR \
    wandb_dir=./wandb \
    disable_wandb=true \
    run_name=smoke_high_20step_2gpu \
    max_steps=20 \
    sample_at_steps='[1]' \
    sample_interval=2
