#!/bin/bash
#SBATCH --job-name=z-critic-v6b-16gpu
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ej/as1748.u6ej/ARRWM/logs/%x_%j.err

# =====================================================================
# v6 — z-predictor critic (replaces reward critic from v3-v5):
#
# Critic overhaul – predict full 8D z:
#   - Critic output changed from scalar reward to 8D ss_vae z latent
#   - Trained with weighted MSE: 2x weight on z2/z7 (action dims)
#   - Each chunk now has a unique 8D target (diverse gradients)
#   - Removed: reward formula, asymmetric diff, log/exp transforms
#
# Generator guidance – z-based (v6b fix):
#   - Guidance only on z2/z7 (generator has no control over other dims)
#   - Target = 1.1 * commanded z2/z7 (slight overshoot to counter undershoot)
#   - Minimises MSE(critic_pred_z2z7, 1.1 * commanded) through frozen critic
#   - generator_action_z_guidance_weight = 0.1
#   - 500-step linear warmup after LR warmup
#
# Unchanged from v5:
#   - 4 nodes x 4 GPUs = 16 GPUs
#   - LoRA rank 256, alpha 256
#   - Generator lr 5e-5, critic lr 3e-4
#   - streaming_chunk_size 21, context_frames 3
#   - Critic arch: base_channels 128, 4 res blocks, SE, AttentionPool
# =====================================================================

set -e -o pipefail

cd /scratch/u6ej/as1748.u6ej/ARRWM

CONFIG=configs/causal_lora_diffusion_teacher.yaml
LOGDIR=/scratch/u6ej/as1748.u6ej/ARRWM/logs/z_critic_v6b
WANDB_SAVE_DIR=wandb

source /scratch/u6ej/as1748.u6ej/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ej/as1748.u6ej/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

mkdir -p "$LOGDIR"
mkdir -p logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CONFIG=$CONFIG"
echo "Starting z-critic v6b training on $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  train.py \
  --config_path $CONFIG \
  --logdir $LOGDIR \
  --wandb-save-dir $WANDB_SAVE_DIR \
  --no-auto-resume

echo "Training completed on $(date)"
