#!/bin/bash
#SBATCH --job-name=v14-aoo-smoke
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Study 1 (AOO) dual-view SMOKE: forward + rear (reverse) modelled
# simultaneously by the SAME v14 model. 5 iters, 1 GPU, no eval/ckpt.
# PRECONDITION: rear zarrs + <ride>.align.npy must exist under
# /projects/u6ex/fbots/frodobots_encoded_weu_rear (encode-rear-smoke +
# build_rear_alignment). Do NOT submit before alignment is confirmed.
# =====================================================================
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/causal_lora_diffusion_teacher_v14_aoo_smoke.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14_aoo_smoke
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR" "$LOGDIR" logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))
export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CONFIG=$CONFIG"
echo "Starting v14-AOO dual-view SMOKE on $(date)"
srun --ntasks-per-node=1 bash -c 'nvidia-smi --query-gpu=index,memory.total,memory.free --format=csv,noheader || true'

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR

echo "v14-AOO smoke completed on $(date)"
