#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-14B-teacher-G-v3
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-3 14B-teacher G v3: identical to v2 (j4627383), with the SOLE
# change being ``anti_collapse_mean_target=gt`` (replacing v2's implicit
# default ``"zero"``).
#
# Why: v2 cancelled. v2's mean anchor was ``mean(pred_x0)^2`` —
# pulled per-(batch, frame) channel-pooled pred mean toward 0. That
# assumed the VAE's latent prior is zero-centered, which only holds
# in aggregate; per-frame GT means are NOT zero, and pulling pred
# toward zero biases the prediction away from the GT frame's natural
# offset. v3 switches the anchor target from 0 to GT's per-(batch,
# frame) mean:
#
#   v2: loss_mean = mean(pred_x0).pow(2).mean()                # pred -> 0
#   v3: loss_mean = (mean(pred_x0) - mean(GT).detach()).pow(2).mean()  # pred -> GT
#
# Both forms are still SCALAR-ONLY per (batch, frame) — one number
# per frame, NOT pixel-wise supervision. The GT mean is detached so
# no gradient flows back through it; this is purely a target for the
# pred mean to anchor onto. gt_latent_mse/mae remain OFF — we still
# do not want full pixel-wise GT supervision (which would override
# DMD), only the per-frame mean offset.
#
# The std-floor term (``anti_collapse_loss_weight=1.0``) is unchanged:
# still ``ReLU(std(GT) - std(pred_x0))^2`` (gray-collapse guard).
# Combined with the GT-anchored mean, the latent now has both its
# spread (std >= GT std) and its location (mean ~= GT mean)
# constrained on a per-frame basis, without any pixel-wise loss.
#
# Code change accompanying v3 (in model/dmd_action_forcing.py):
# new knob ``anti_collapse_mean_target`` (default "zero" for v2/back-
# compat; v3 sets it to "gt"). The std-floor is unaffected.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

# ---- SAM2 paths (required when gan_enabled=true) ----
GAN_SAM2_CKPT=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_base_plus.pt
GAN_SAM2_CFG=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_b+.yaml

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

# Pre-flight SAM2 sanity checks (fail-fast on missing prereqs).
python -c "import sam2" 2>/dev/null || { echo "ERROR: sam2 package not installed."; exit 1; }
[ -f "$GAN_SAM2_CKPT" ] || { echo "ERROR: SAM2 checkpoint not found at $GAN_SAM2_CKPT"; exit 1; }
[ -f "$GAN_SAM2_CFG" ] || { echo "ERROR: SAM2 config not found at $GAN_SAM2_CFG"; exit 1; }

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_14B_teacher_G_v3
WANDB_SAVE_DIR=wandb

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:512

echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "Starting Phase-3 14B-TEACHER G v3 (v2 + mean anchor target=gt) on $(date)"
echo "  * real_score: Wan2.1-T2V-14B FROZEN (no v14 LoRA)."
echo "  * fake_score: Wan2.1-T2V-1.3B (default)."
echo "  * generator: 1.3B causal student (default)."
echo "  * 21-frame student rollout, 200-step max."
echo "  * GAN ON (sam2_pixel, res=512), MANIQA OFF, aux teacher OFF, online teacher OFF."
echo "  * anti_collapse_loss_weight=1.0 (std floor, gray-collapse guard)."
echo "  * anti_collapse_mean_weight=1.0 (mean anchor)."
echo "  * anti_collapse_mean_target=gt (NEW — pred mean anchored to GT"
echo "    per-(batch, frame) mean, not 0)."

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
    wandb_project=longlive-phase-3 \
    max_steps=200 \
    real_name=Wan2.1-T2V-14B \
    fake_name=Wan2.1-T2V-1.3B \
    model_kwargs.model_name=Wan2.1-T2V-1.3B \
    v14_teacher_checkpoint=null \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    max_rolls_per_ride=300 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=21 \
    num_workers=0 \
    boundary_vae_roundtrip=false \
    local_attn_size_schedule=[[0,21]] \
    dmd_context=self \
    real_teacher_train_online=false \
    real_teacher_causal_mask=false \
    real_guidance_scale=3.0 \
    min_score_timestep=40 \
    max_score_timestep=44 \
    aux_teacher_loss_weight=0.0 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    warm_start_init=false \
    gan_enabled=true \
    gan_loss_weight=1.0 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=512 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=10 \
    gan_warmup_steps=80 \
    gan_sam2_frame_pool=none \
    gan_sam2_encoder_chunk_size=6 \
    gan_critic_dense_distillation=true \
    flash_dmd_enabled=false \
    maniqa_approx_loss_weight=0.0 \
    dmd_normalization_enabled=false \
    anti_collapse_loss_weight=1.0 \
    anti_collapse_mean_weight=1.0 \
    anti_collapse_mean_target=gt \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=action_forcing_phase3_14B_teacher_G_v3_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
