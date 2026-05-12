#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-G-NR-v5-no-streaming
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
# Phase-3 online_teacher_G_NR_v4_no_streaming: same as v3 but adds a
# decaying GT-mix on the DMD-scoring clean_x.
#
# v4 DELTA vs v3
# --------------
# At the start of training, the teacher's scoring clean_x context is
# 30% noised-GT + 70% student-cache_pred. This linearly decays to 0%
# GT (pure student-cache_pred) over the first 100 outer steps. After
# step 100, identical to v3's clean_x behavior.
#
# Why a brief GT-anchored start helps:
#   * Steps [0, 100) are the LoRA's bootstrap window — the LoRA is
#     still ramping up (loss weight reaches plateau at step 100), so
#     the teacher's own internal context isn't yet well-adapted to
#     student-distribution clean_x. Mixing in a small amount of GT
#     gives the teacher a stable scoring context while the LoRA is
#     still learning the bridge function.
#   * By step 100, the LoRA pointing-vehicle has plateaued — the
#     teacher can rely on pure self-context, and DMD scoring
#     transitions to the fully student-driven regime that matches the
#     deployment-time use case.
#
# Mechanism (existing infrastructure, see _apply_dmd_context_mix_p_
# schedule at trainer/causal_action_forcing_train.py:5496-5564):
#   * dmd_context              = mix           (was self in v3)
#   * dmd_context_mix_p        = 0.30          (start; ramp begins at step 0)
#   * dmd_context_mix_p_target = 0.00          (end of ramp)
#   * dmd_context_mix_p_ramp_steps = 100       (linear interpolation)
#
# Schedule:
#   step  0  → p = 0.30
#   step 50  → p = 0.15
#   step 100 → p = 0.00 (and held thereafter)
#
# Affects ONLY the DMD-scoring clean_x_real path:
#   sc_clean_x_real = (1 - p) * sc_clean_x + p * noised_GT
# The aux pass's own clean_x (built separately from
# [GT-seed | student-cache-snapshot] in compute_aux_teacher_loss_
# streaming) is NOT touched by this knob.
#
# All v3 design choices preserved — see header in v3 sbatch for the
# full rationale on the LoRA-as-pointing-vehicle setup, GAN-side
# stability tuning, and the aux/DMD/GAN ramp alignment.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_G_NR_v5_no_streaming
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
echo "Starting Phase-3 online-teacher G_NR v5 (no streaming) on $(date)"
echo "  * 21-frame iter-1-only student rollout, 200-step max."
echo "  * max_rolls_per_ride=1 — streaming resets every step (no iter k>=2)."
echo "  * Teacher: fresh rank-32 LoRA on v14-merged base (alpha=64, r=32)."
echo "  * Steps [0, 20): LoRA frozen. Steps [20, 100): aux loss weight ramps"
echo "    0 -> 1.0. Steps [100, 200): LoRA pointing-vehicle at plateau."
echo "  * GT-mix in DMD clean_x: 0.30 -> 0.00 linearly over first 100 steps."
echo "  * Student first-10-steps signal: MANIQA only (DMD off, GAN critic-only)."
echo "  * gan_loss_weight=0.25, gan_critic_warmup=10, gan_warmup=200"
echo "    (pure ramp across A/B horizon, terminal effective weight 0.2375)."
echo "  * dmd_loss_start_step=50, dmd_loss_warmup_steps=100 (full at step 150)."
echo "    DMD reaches full strength 50 steps after LoRA plateau."
echo "  * aux_teacher_send_student_grad=false (DMD is sole student-grad path"
echo "    from the teacher; aux LoRA optim is severed from the student)."
echo "  * real_teacher_input_source=student, real_teacher_causal_mask=false"
echo "    (LoRA learns bidirectional student->GT bridge)."
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
    dmd_context=mix \
    dmd_context_mix_p=0.30 \
    dmd_context_mix_p_target=0.00 \
    dmd_context_mix_p_ramp_steps=100 \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=1.0 \
    real_teacher_lr=5e-5 \
    real_teacher_warmup_steps=50 \
    real_teacher_input_source=student \
    teacher_cadence=fake \
    aux_teacher_p_schedule_enabled=false \
    aux_teacher_send_student_grad=false \
    aux_teacher_start_step=20 \
    aux_teacher_loss_weight=1.0 \
    aux_teacher_loss_warmup_steps=80 \
    aux_teacher_lora_rank=32 \
    aux_teacher_lora_alpha=64 \
    dmd_loss_start_step=50 \
    dmd_loss_warmup_steps=100 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    gan_loss_weight=0.05 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=768 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=10 \
    gan_warmup_steps=100 \
    gan_sam2_frame_pool=none \
    gan_sam2_encoder_chunk_size=84 \
    maniqa_approx_loss_weight=1.0 \
    maniqa_n_frames=84 \
    maniqa_n_patches_per_frame=8 \
    maniqa_metric_name=maniqa-pipal \
    gan_critic_dense_distillation=true \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=action_forcing_phase3_online_teacher_G_NR_v5_no_streaming_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
