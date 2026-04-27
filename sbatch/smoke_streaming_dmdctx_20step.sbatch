#!/bin/bash
#SBATCH --job-name=smoke-streaming-dmdctx-20step
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# 4-rank DDP streaming-mode smoke (20 steps), post-fix Phase-B path.
# Reuses the cached 16-ride manifest in
# logs/diag_streaming/test2_ddp4gpu/.ride_manifest.pt to skip the
# 858-zarr scan on startup (~3 min saved).
#
# Pass criteria (manual):
#   - 20 steps, no deadlock, no NaN, no `sequence exhausted`
#   - "loaded 16 rides (no scan)" appears once
#   - sample mp4s saved at step 1 and at step 11 (next gen-step
#     after step 10) under <log_dir>/<run_name>/samples/
#   - "Training complete (step=20)" prints

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

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
export TOKENIZERS_PARALLELISM=false
export WANDB_MODE=disabled

export ARRWM_MANIFEST_PICKLE=/scratch/u6ex/as1748.u6ex/ARRWM/logs/diag_streaming/test2_ddp4gpu/.ride_manifest.pt

LOG_DIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_streaming_dmdctx_j${SLURM_JOB_ID}
mkdir -p "$LOG_DIR"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

echo "=== $(date) job=${SLURM_JOB_ID} node=$(hostname) ==="
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader

srun --ntasks=1 --gpus-per-node=4 torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_action_forcing_train.py \
  --config configs/action_forcing_phase1_aux_dmdctx.yaml \
  --override \
    streaming_mode=true \
    max_steps=20 \
    disable_wandb=true \
    log_dir=$LOG_DIR \
    log_interval=1 \
    checkpoint_interval=999 \
    auto_resume=false

echo "=== smoke run complete, see $LOG_DIR ==="
