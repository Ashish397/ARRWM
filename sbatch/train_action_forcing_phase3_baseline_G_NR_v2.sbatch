#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-baseline-G-NR-v2
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-3 baseline_G_NR_v2: same configuration as baseline_G_NR but
# with GAN-side stability tuning after the v1 run (slurm 4513839 /
# wandb run G_NR) destabilised around step 46.
#
# v1 trace summary:
#   * step 31 (effective G-side GAN weight ~0.39): healthy, c_corr=-0.49
#   * step 41 (effective ~0.51): c_corr flipped to +0.63 → critic
#     collapsing onto D
#   * step 46+ : D real/fake separation lost, gen_grad jumped 10×,
#     fake_grad reached 17, gen_loss went negative — full GAN collapse.
#
# Empirical stable ceiling for effective G-side GAN weight ≈ 0.40,
# with collapse-onset observed around 0.51. v2 stretches the warmup
# across the entire 200-step A/B horizon so the signal builds slowly,
# only reaching the 0.50 plateau target at the final step.
#
# Changes vs v1:
#   * gan_loss_weight     1.0 → 0.50  (terminal weight ≈ v1 collapse-
#                                     onset, but reached gradually)
#   * gan_warmup_steps     80 → 200   (warmup spans the full A/B
#                                     horizon; no plateau training,
#                                     monotonic ramp throughout)
#   * gan_critic_warmup_steps unchanged at 10 — generator is pretrained
#     so the SAM2 critic can come up fast without destabilising.
#
# Effective G-side GAN weight under v2:
#   step  31 → 0.078   (v1 was 0.39)
#   step  46 → 0.115   (v1 was 0.58 — collapse point)
#   step  80 → 0.200   (v1 was 1.00)
#   step 120 → 0.300
#   step 200 → 0.500   (terminal; matches v1's collapse-onset weight
#                       but the system has had 200 steps to adapt to
#                       each intermediate level rather than 41)
#
# Early-warning to watch: c_corr. In v1 it crossed zero at step 41,
# one block before the visible collapse. If c_corr stays negative
# through step 120 here, the headroom is real.
#
# All other v1 settings preserved (DMD / MANIQA / SAM2 wiring,
# frozen v14 teacher, no aux loss).
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_baseline_G_NR_v2
WANDB_SAVE_DIR=wandb

# ---- SAM2 paths (must be set; see PREREQUISITES) ----
GAN_SAM2_CKPT=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_base_plus.pt
GAN_SAM2_CFG=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_b+.yaml

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

# Pre-flight SAM2 sanity checks. Fail-fast on missing prereqs so the
# job exits at submission rather than ~5 min into model load.
python -c "import sam2" 2>/dev/null || {
  echo "ERROR: sam2 package not installed. See PREREQUISITES in this sbatch."
  exit 1
}
[ -f "$GAN_SAM2_CKPT" ] || {
  echo "ERROR: SAM2 checkpoint not found at $GAN_SAM2_CKPT"
  exit 1
}
[ -f "$GAN_SAM2_CFG" ] || {
  echo "ERROR: SAM2 config not found at $GAN_SAM2_CFG"
  exit 1
}

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
echo "GAN_SAM2_CKPT=$GAN_SAM2_CKPT"
echo "GAN_SAM2_CFG=$GAN_SAM2_CFG"
echo "Starting Phase-3 BASELINE G_NR v2 on $(date)"
echo "  * 21-frame student rollout, 200-step max."
echo "  * Teacher: v14 LoRA merged into base WAN, FROZEN. No online"
echo "    teacher training, no aux loss."
echo "  * GAN stability tune: gan_loss_weight=0.20, gan_warmup_steps=200"
echo "    (warmup spans full A/B horizon), gan_critic_warmup_steps=10"
echo "    (pretrained G → critic can ramp fast)."
echo "  * wandb_project=longlive-phase-3."

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
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    max_rolls_per_ride=1 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=21 \
    num_workers=0 \
    boundary_vae_roundtrip=true \
    local_attn_size_schedule=[[0,21]] \
    dmd_context=self \
    real_teacher_train_online=false \
    real_teacher_causal_mask=false \
    real_guidance_scale=1.0 \
    aux_teacher_loss_weight=0.0 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    gan_loss_weight=0.2 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=768 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=10 \
    gan_warmup_steps=200 \
    gan_sam2_frame_pool=none \
    gan_sam2_encoder_chunk_size=84 \
    maniqa_approx_loss_weight=1.0 \
    maniqa_n_frames=84 \
    maniqa_n_patches_per_frame=8 \
    maniqa_metric_name=maniqa-pipal \
    gan_critic_dense_distillation=true \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=action_forcing_phase3_baseline_G_NR_v2_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
