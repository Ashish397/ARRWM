#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-NR-G-v27F
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
# Phase-3 online_teacher_NR_G_v27F: v27E + GAN FULLY DISABLED +
# max_gradient_chunks=3 -> 5. With GAN off, the SAM2 backbone, R3GAN
# disc heads, LatentSAM2Critic, and all perceptual-approx forwards
# (MANIQA, LPIPS, MS-SSIM, MSE-approx — bundled inside the GAN path)
# are SKIPPED at __init__ via the gan_enabled=false short-circuit.
# That recovers ~15-25 GB of GPU memory, which is spent on the extra
# 2 chunks of gradient-on student activations.
#
# v27F vs v27E — knob deltas
# --------------------------
#   gan_enabled            : - (default false) -> false   (EXPLICIT)
#   max_gradient_chunks    : 3                 -> 5
# Effective wandb signals dropped: train/r3gan_*, train/perc_loss_*,
# critic head training, MANIQA distillation. DMD + anti_collapse +
# aux_teacher remain — these are now the only gen-side signals.
#
# WHY DISABLE GAN ENTIRELY
# ------------------------
# wandb diagnostic from v27D (steps 40-90): the gen-side GAN
# contribution (g_weighted ≈ -0.0007) was ~800× smaller than
# perc_loss_weighted (-0.544). The GAN knob has effectively no
# authority over the student at v27 settings — disabling it costs
# essentially nothing in gen-side training signal, and recovers
# substantial memory. The c_corr saturation seen at step 131 of v27D
# is a ranking-metric artifact of an ineffective disc, not a real
# collapse driver.
#
# v27E POST-PROCESS (RETAINED)
# ----------------------------
#   forward_noiser_lowfreq_smooth_window = 3 (length-3 boxcar)
#   forward_noiser_lowfreq_alpha         = 1.0
# clean_x_aux = gt + alpha * lowpass(fn - gt). Same as v27E.
#
# WHY THE REPLACEMENT
# -------------------
# Architecture review of the v27D forward noiser (4 ResBlocks of
# factorized 3D conv: spatial 1x3x3 + temporal 3x1x1, 9x9 spatial RF,
# 9-frame temporal RF, no down/upsampling, no attention,
# residual+zero-init out) concluded:
#   * High-freq capture: GOOD.
#   * Spatially-varying / across-many-chunks low-freq drift: WEAK.
# clean_x_aux looks high-freq-accurate but low-freq is hard to see.
#
# v27E POST-PROCESS (no extra params, no autograd path; no_grad block):
#
#   residual = fn_causal_AR_GT - gt_target          (FN-added noise)
#   lowfreq  = temporal_lowpass(residual, window=3) (boxcar along F)
#   replaced = gt_target + alpha * lowfreq
#
# Length-3 boxcar has nulls at f=1/3 and 2/3, exactly nulling the
# per-chunk period-3 high-frequency element identified by the user
# while preserving anything period > 3 (the across-chunk drift). The
# result is GT + smoothed noise only — no FN high-freq passes
# through. window=0 disables the post-process (no-op).
#
# All other v27D content preserved below.
#
# =====================================================================
# LEGACY v27D context: same v27 chain — see v27D/v27B/v27 sbatches for
# the full architecture motivation (v21 alt-head pathology,
# rollout1->rollout2 mapping, etc.). v27E only adds the post-process
# boost on top of v27D's running training contract.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_NR_G_v27F
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.6,max_split_size_mb:512

echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "GAN_SAM2_CKPT=$GAN_SAM2_CKPT"
echo "GAN_SAM2_CFG=$GAN_SAM2_CFG"
echo "Starting Phase-3 online-teacher NR_G v27F (v27E + GAN OFF + max_grad_chunks=5) on $(date)"
echo "  * v27F = v27E with two deltas:"
echo "      gan_enabled         = false  (NEW; SAM2 + critic + heads SKIPPED at init)"
echo "      max_gradient_chunks = 3 -> 5"
echo "  * Memory budget: GAN-off frees ~15-25 GB (SAM2 backbone + perc"
echo "    approx forwards + critic), spent on 2 extra grad chunks."
echo "  * No gen-side GAN/perc signal — DMD + anti_collapse + aux_teacher"
echo "    are the only gen-side gradients."
echo "  * forward_noiser low-freq smoothing (v27E) RETAINED: window=3, alpha=1.0."

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
    aux_teacher_loss_weight=0.5 \
    aux_teacher_loss_warmup_steps=40 \
    aux_teacher_lora_rank=32 \
    aux_teacher_lora_alpha=64 \
    dmd_loss_weight=0.5 \
    dmd_loss_start_step=0 \
    fake_alt_head_enabled=true \
    fake_alt_head_start_step=0 \
    fake_alt_apply_start_step=0 \
    fake_alt_target_mode=rollout2_student \
    fake_alt_rollout2_num_seed_chunks=1 \
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
    dmd_loss_warmup_steps=80 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    real_score_ema_weight=0.0 \
    gan_enabled=false \
    gan_loss_weight=0.0 \
    gan_lr=5e-5 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=768 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=40 \
    gan_disc_start_step=40 \
    gan_warmup_steps=25 \
    gan_warmup_shape=linear \
    perceptual_approx_warmup_steps=0 \
    perceptual_approx_ramp_steps=15 \
    gan_updates_per_step=1 \
    gan_r1_gamma=2.0 \
    gan_r2_gamma=2.0 \
    gan_sam2_frame_pool=none \
    gan_sam2_encoder_chunk_size=84 \
    maniqa_approx_loss_weight=1.0 \
    maniqa_n_frames=84 \
    maniqa_n_patches_per_frame=8 \
    maniqa_metric_name=maniqa-pipal \
    gan_critic_dense_distillation=true \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    gt_latent_mse_loss_weight=0.0 \
    gt_latent_mae_loss_weight=0.0 \
    anti_collapse_loss_weight=0.5 \
    anti_collapse_mean_weight=0.05 \
    max_gradient_chunks=5 \
    run_name=action_forcing_phase3_online_teacher_NR_G_v27F_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
