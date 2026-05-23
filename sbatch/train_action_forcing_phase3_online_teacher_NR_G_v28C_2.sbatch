#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-NR-G-v28C_2
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
# Phase-3 online_teacher_NR_G_v28C: minimal modern stack.
#
# Pieces ON:
#   * Online DMD (standard, with anti-collapse std-floor guard).
#   * Aux teacher (online LoRA training of real_score against GT).
#   * LADD GAN with WAVELET-HF disc input (WGSR-style: disc sees only
#     HF latent sub-bands; LL drops out so the GAN gradient can only
#     push HF detail and never pulls content around — DMD handles
#     content anchoring).
#   * GAN pair mode: ``gt_vs_fake`` (LADD canonical). The disc sees
#     gen chunks vs GT chunks at the same temporal positions.
#
# Pieces OFF:
#   * Forward noiser (CAR-noiser of v27B/F/H). DISABLED — we want a
#     clean baseline where the LADD wavelet GAN is the sole detail
#     supervisor (besides DMD's content anchor).
#   * MANIQA NR-IQA perceptual approx. DISABLED — same reasoning.
#   * fake_alt head + rollout-2 prebuild. DISABLED — no consumer.
#   * ASD-style adjacent-chunk pair loss. DISABLED — v28B turns it on.
#
# Knob deltas vs v28 (LADD-with-everything)
# -----------------------------------------
#   forward_noiser_enabled           true  -> false
#   maniqa_approx_loss_weight        1.0   -> 0.0
#   fake_alt_head_enabled            true  -> false
#   aux_real_clean_x_source          causal_AR_GT -> self
#       (causal_AR_GT requires the forward noiser; revert to ``self`` —
#       use the streaming-state clean ref like v27 and earlier.)
#   ladd_wavelet_hf_enabled          false -> true   (NEW)
#   ladd_wavelet_hf_drop_ll          -     -> true   (NEW)
#   ladd_gt_vs_fake_enabled          false -> true   (NEW pair mode)
#   ladd_adjacent_chunks_enabled     true  -> false  (NEW pair mode)
#
# All other v28 knobs preserved.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_NR_G_v28C
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
echo "Starting Phase-3 online-teacher NR_G v28C (LADD wavelet-HF + DMD + aux) on $(date)"
echo "  * GAN: LADD teacher-feature, gt_vs_fake, wavelet HF input."
echo "  * DMD: online, anti-collapse on, no forward noiser."
echo "  * Aux teacher: online LoRA training of real_score against GT."
echo "  * OFF: forward noiser, MANIQA, alt-head, adjacent-chunk loss."

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
    aux_real_clean_x_source=gt \
    dmd_loss_weight=0.25 \
    dmd_loss_start_step=20 \
    dmd_loss_warmup_steps=150 \
    dmd_normalization_enabled=false \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    real_score_ema_weight=0.0 \
    fake_alt_head_enabled=false \
    forward_noiser_enabled=false \
    forward_noiser_apply_in_aux=false \
    forward_noiser_lowfreq_smooth_window=0 \
    critic_clean_x_source=self \
    gan_enabled=true \
    gan_backbone=ladd_teacher_feat \
    gan_loss_weight=0.01 \
    gan_lr=2e-5 \
    gan_critic_warmup_steps=40 \
    gan_disc_start_step=40 \
    gan_warmup_steps=100 \
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
    ladd_r1_every_n_steps=10 \
    ladd_diff_aug_policy=flip,cutout,translation \
    ladd_pairs_per_step=2 \
    ladd_wavelet_hf_enabled=true \
    ladd_wavelet_hf_drop_ll=true \
    ladd_wavelet_hf_adapter_init_gain=0.1 \
    ladd_gt_vs_fake_enabled=true \
    ladd_gt_vs_fake_weight=1.0 \
    ladd_adjacent_chunks_enabled=false \
    ladd_adjacent_chunks_weight=0.0 \
    perceptual_approx_warmup_steps=0 \
    perceptual_approx_ramp_steps=15 \
    maniqa_approx_loss_weight=0.0 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    gt_latent_mse_loss_weight=0.0 \
    gt_latent_mae_loss_weight=0.0 \
    anti_collapse_loss_weight=0.8 \
    anti_collapse_mean_weight=0.0 \
    max_gradient_chunks=5 \
    run_name=action_forcing_phase3_online_teacher_NR_G_v28C_2_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
