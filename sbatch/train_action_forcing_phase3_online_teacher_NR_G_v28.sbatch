#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-NR-G-v28
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
# Phase-3 online_teacher_NR_G_v28: swap SAM2-pixel GAN out for the
# LADD-style teacher-feature discriminator with adjacent-chunk
# adversarial alignment.
#
# v28 vs v27B — knob deltas
# -------------------------
# REMOVED (SAM2-specific, ignored under gan_backbone=ladd_teacher_feat):
#   gan_sam2_*  (checkpoint, config, resolution, frame_pool, etc.)
#   gan_sam2_distilled_critic
#   gan_critic_*
#   gan_critic_dense_distillation
#
# ADDED (LADD adjacent-chunk disc):
#   gan_backbone                       sam2_pixel -> ladd_teacher_feat
#   ladd_feature_blocks                []        (auto: 5 taps over WAN depth)
#   ladd_proj_dim                      256
#   ladd_use_csm                       true      (Projected-GAN FPN fusion)
#   ladd_use_prompt_cond               true      (StyleGAN-T cmap)
#   ladd_cmap_dim                      64
#   ladd_disc_loss_weight              1.0       (extra G-side multiplier)
#   ladd_r1_gamma                      1.0
#   ladd_diff_aug_policy               "flip,cutout,translation"
#   ladd_pairs_per_step                0         (use all adjacent pairs)
#
# All other v27B knobs preserved (CAR noiser + alt head + MANIQA + DMD
# + flash-DMD t=60 stay on). MANIQA build path is now independent of
# the GAN gate (perceptual-approx block moved out of `if gan_enabled:`)
# so MANIQA continues to fire under the LADD disc.
#
# What LADD changes architecturally
# ---------------------------------
# - The SAM2 pixel-disc + LatentSAM2Critic stack is GONE.
# - A small (~10-15M trainable) head stack sits on top of the EXISTING
#   real_score teacher: features tapped at 5 WAN transformer block
#   indices (auto-default for 1.3B = [3,10,17,24,29]); CCM 1x1 channel
#   mixing; FPN-style cross-scale CSM fusion; per-tap 1D SpectralConv
#   heads.
# - Adversarial pair = ADJACENT CHUNKS in the gen-side rollout:
#   chunk_i (less AR drift) as real anchor; chunk_{i+1} (more AR drift)
#   as fake / push-target. RpGAN loss. Adapts ASD's "smoother
#   distribution-level alignment" principle from adjacent denoising
#   steps to adjacent rollout chunks.
# - DiffAugment-equivalent applied identically to real and fake at
#   latent level (flip / cutout / translation) for disc regularisation.
# - flash-DMD t=60 forward output feeds the disc when flash on (same
#   source as v27B); falls back to pred_image at t=0 when flash off.
#
# Memory: net save vs v27B on 1.3B teacher (no SAM2 encoder, no VAE
# pixel decode for disc). +1 extra WAN teacher forward per gen step
# for the disc — within budget at 95GB H100.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_NR_G_v28
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "Starting Phase-3 online-teacher NR_G v28 (LADD adjacent-chunk disc) on $(date)"
echo "  * GAN backbone: LADD (frozen WAN-teacher features + trainable heads)."
echo "  * Pair source: adjacent chunks in rollout 1 (chunk_i / chunk_{i+1})."
echo "  * Disc input timestep: flash_dmd_gan_t=60 when flash-DMD on (it is)."
echo "  * DiffAugment policy: flip,cutout,translation."
echo "  * CAR noiser + alt-head + MANIQA + DMD all carried over from v27B."
echo "  * SAM2 path REMOVED — model/ladd_disc.py + the build branch in"
echo "    trainer/causal_action_forcing_train.py provide the new disc."
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
    max_steps=250 \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    max_rolls_per_ride=1 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=21 \
    num_workers=0 \
    sample_fps=20 \
    boundary_vae_roundtrip=true \
    local_attn_size_schedule=[[0,21]] \
    dmd_context=self \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=0.0 \
    lr=2e-5 \
    real_teacher_lr=7.5e-5 \
    real_teacher_warmup_steps=0 \
    real_teacher_input_source=gt \
    teacher_cadence=fake \
    aux_teacher_p_schedule_enabled=false \
    aux_teacher_send_student_grad=false \
    aux_teacher_start_step=0 \
    aux_teacher_loss_weight=1.0 \
    aux_teacher_loss_warmup_steps=80 \
    aux_teacher_lora_rank=32 \
    aux_teacher_lora_alpha=64 \
    dmd_loss_weight=0.5 \
    dmd_loss_start_step=20 \
    fake_alt_head_enabled=true \
    fake_alt_head_start_step=0 \
    fake_alt_apply_start_step=0 \
    fake_alt_target_mode=rollout2_student \
    fake_alt_rollout2_num_seed_chunks=2 \
    forward_noiser_enabled=true \
    forward_noiser_hidden_dim=512 \
    forward_noiser_num_blocks=4 \
    forward_noiser_max_carn_step=8 \
    forward_noiser_loss_weight=1.0 \
    forward_noiser_apply_in_aux=true \
    forward_noiser_lr=5e-5 \
    critic_clean_x_source=self \
    aux_real_clean_x_source=causal_AR_GT \
    dmd_normalization_enabled=false \
    dmd_loss_warmup_steps=150 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    real_score_ema_weight=0.0 \
    gan_enabled=true \
    gan_backbone=ladd_teacher_feat \
    gan_loss_weight=0.2 \
    gan_lr=5e-5 \
    gan_critic_warmup_steps=40 \
    gan_disc_start_step=40 \
    gan_warmup_steps=25 \
    gan_warmup_shape=quadratic \
    gan_updates_per_step=1 \
    gan_r1_gamma=2.0 \
    gan_max_grad_norm=10.0 \
    ladd_proj_dim=256 \
    ladd_use_csm=true \
    ladd_use_lateral_proj=false \
    ladd_use_prompt_cond=true \
    ladd_cmap_dim=64 \
    ladd_disc_loss_weight=1.0 \
    ladd_r1_gamma=1.0 \
    ladd_diff_aug_policy=flip,cutout,translation \
    ladd_pairs_per_step=0 \
    perceptual_approx_warmup_steps=0 \
    perceptual_approx_ramp_steps=15 \
    maniqa_approx_loss_weight=1.0 \
    maniqa_n_frames=84 \
    maniqa_n_patches_per_frame=8 \
    maniqa_metric_name=maniqa-pipal \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    gt_latent_mse_loss_weight=0.0 \
    gt_latent_mae_loss_weight=0.0 \
    anti_collapse_loss_weight=0.0 \
    anti_collapse_mean_weight=0.0 \
    max_gradient_chunks=0 \
    run_name=action_forcing_phase3_online_teacher_NR_G_v28_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
