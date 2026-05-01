#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-bidir-baseline-3h-8node
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing DMD — BIDIR BASELINE, 8 nodes x 4 GPUs, 3h.
#
# Pure DMD baseline (aux OFF, no R3GAN, no SC-DMD) bound to
# configs/action_forcing_phase1_baseline.yaml — same as the
# baseline sbatch with TWO deltas:
#   * real_teacher_causal_mask=false — v14 real_score's TF joint
#     forward runs FULLY BIDIRECTIONAL (no causal block_mask on the
#     joint clean+noisy sequence). The DMD bidir scorer joint
#     forward is already unmasked in baseline; this flag flips the
#     v14-LoRA teacher's TF mask too, so cond/uncond forwards see
#     all 21 noisy frames jointly with no left-only attention.
#   * real_guidance_scale=1.0 — milder CFG (cond + 1*(cond-uncond)
#     = 2*cond - uncond), down from baseline's 3.0 (4*cond - 3*uncond).
#     NOTE: scale=1.0 is NOT "CFG off"; set 0.0 if you want raw cond.
#
# Sister sbatches:
#   sbatch/train_action_forcing_phase1_baseline_16h_8node.sbatch (causal teacher, real_guidance_scale=3.0)
#   sbatch/train_action_forcing_phase1_gan_16h_8node.sbatch      (baseline + R3GAN)
#   sbatch/train_action_forcing_phase1_scdmd_16h_8node.sbatch    (baseline + SC-DMD)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_baseline.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_bidir_baseline_3h_8node
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
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
echo "LOGDIR=$LOGDIR"
echo "Starting Phase-1 Action-Forcing DMD bidir baseline (8 nodes x 4 GPUs, 3h) on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true
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
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$LOGDIR \
    wandb_dir=$WANDB_SAVE_DIR \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    max_rolls_per_ride=300 \
    dfake_gen_update_ratio=5 \
    collapse_mae_threshold=0.1 \
    mae_extension_threshold=0.1 \
    model_kwargs.local_attn_size=42 \
    local_attn_size_schedule=[[0,42]] \
    dmd_context=self \
    real_teacher_train_online=false \
    real_teacher_causal_mask=false \
    real_guidance_scale=1.0 \
    num_workers=0 \
    run_name=action_forcing_phase1_bidir_baseline_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
