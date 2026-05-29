#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-NR-G-v28G-4
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-3 online_teacher_NR_G_v28E-6
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_NR_G_v28D
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
echo "Starting Phase-3 online-teacher NR_G v28G-4 (flash_dmd OFF + aux disc feat-match) on $(date)"
echo "  * FLASH-DMD DISABLED: no extra t=60 grad-on forward per block."
echo "    GAN gen-side adv path falls back to pred_image (random-exit"
echo "    rung) instead of flash_dmd_gan_x0. Disc input stays clean"
echo "    (wavelet_hf forces disc_t_int=0)."
echo "    With recent pipeline changes (clean_chunk unconditional +"
echo "    fake_alt rollout-2 prefer_cache_pred_in_output=True), aux"
echo "    teacher and fake_alt rollout-2 still receive the cleanest"
echo "    available x0 estimate (t~178.6 finish-denoised cache_pred)."
echo "  * AUX TEACHER FEATURE-MATCHING: aux_teacher_disc_feat_weight=0.1."
echo "    Borrows the LADD disc's projector features (intermediate WAN"
echo "    teacher activations) as an LPIPS-style supervision channel"
echo "    on the LoRA's x0 estimate, alongside the flow-MSE. L2"
echo "    between projector features of lora_x0 (grad) and gt_target"
echo "    (no_grad). No adversarial framing, no disc-collapse risk."
echo "    Gated by aux_teacher_disc_warmup_steps=100."
echo "  * GAN: LADD teacher-feature, wavelet HF input."
echo "  * Pair modes: gt_vs_fake (w=1.0) AND gt_transition (w=1.0)."
echo "  * DMD: online, anti-collapse on, no forward noiser."
echo "  * Aux teacher: online LoRA training of real_score against GT."

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Clear stale NCCL/CUDA shm segments left by prior killed jobs on each node.
# Only removes files we own. Without this, /dev/shm can fill up and
# NCCL fails with "No space left on device" at process group init.
srun --ntasks-per-node=1 bash -c '
  echo "=== $(hostname) /dev/shm before cleanup ==="
  df -h /dev/shm 2>/dev/null || true
  rm -f /dev/shm/nccl-* /dev/shm/cuda.* /dev/shm/sem.nccl-* 2>/dev/null || true
  echo "=== $(hostname) /dev/shm after cleanup ==="
  df -h /dev/shm 2>/dev/null || true
' || true

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
    sample_fps=16 \
    boundary_vae_roundtrip=false \
    local_attn_size_schedule=[[0,21]] \
    dmd_context=self \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=0.0 \
    lr=1e-5 \
    real_teacher_lr=2.5e-5 \
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
    aux_teacher_disc_adv_weight=0.0 \
    aux_teacher_disc_feat_weight=0.1 \
    aux_teacher_disc_warmup_steps=100 \
    aux_real_clean_x_source=causal_AR_GT \
    dmd_loss_weight=0.25 \
    dmd_loss_start_step=10 \
    dmd_loss_warmup_steps=150 \
    dmd_normalization_enabled=false \
    dmd_frozen_teacher_pass_enabled=false \
    flash_dmd_enabled=false \
    fake_score_ema_weight=0.95 \
    real_score_ema_weight=0.0 \
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
    forward_noiser_lowfreq_smooth_window=0 \
    critic_clean_x_source=self \
    gan_enabled=true \
    gan_backbone=ladd_teacher_feat \
    gan_loss_weight=0.01 \
    gan_lr=2e-5 \
    gan_critic_warmup_steps=20 \
    gan_disc_start_step=20 \
    gan_warmup_steps=100 \
    gan_warmup_shape=quadratic \
    gan_updates_per_step=1 \
    gan_r1_gamma=2.0 \
    gan_max_grad_norm=10.0 \
    ladd_proj_dim=256 \
    ladd_use_csm=true \
    ladd_use_lateral_proj=false \
    ladd_use_prompt_cond=false \
    ladd_cmap_dim=64 \
    ladd_disc_loss_weight=1.0 \
    ladd_r1_gamma=0.01 \
    ladd_r1_every_n_steps=10 \
    ladd_diff_aug_policy=flip,cutout,translation \
    ladd_pairs_per_step=2 \
    ladd_wavelet_hf_enabled=true \
    ladd_wavelet_hf_drop_ll=false \
    ladd_wavelet_hf_ll_weight=0.15 \
    ladd_wavelet_hf_adapter_init_gain=0.1 \
    ladd_gt_vs_fake_enabled=true \
    ladd_gt_vs_fake_weight=1.0 \
    ladd_adjacent_chunks_enabled=false \
    ladd_adjacent_chunks_weight=0.0 \
    ladd_gt_transition_enabled=true \
    ladd_gt_transition_weight=1.0 \
    perceptual_approx_warmup_steps=0 \
    perceptual_approx_ramp_steps=15 \
    maniqa_approx_loss_weight=0.0 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    gt_latent_mse_loss_weight=0.0 \
    gt_latent_mae_loss_weight=0.0 \
    anti_collapse_type=std_mse \
    anti_collapse_loss_weight=0.0 \
    anti_collapse_mean_weight=0.0 \
    stat_anchor_loss_weight=1.0 \
    stat_anchor_STD_short_weight=0.1 \
    stat_anchor_STD_long_weight=0.1 \
    stat_anchor_M2_short_weight=0.1 \
    stat_anchor_M2_long_weight=0.1 \
    stat_anchor_TV_short_weight=0.1 \
    stat_anchor_TV_long_weight=0.1 \
    stat_anchor_rel_tol_short=0.20 \
    stat_anchor_rel_tol_long=0.10 \
    stat_anchor_rampdown_start_step=40 \
    stat_anchor_rampdown_steps=20 \
    max_gradient_chunks=3 \
    run_name=action_forcing_phase3_online_teacher_NR_G_v28G_4_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
