#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-G-NR
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
# Phase-3 online_teacher_G_NR: 21-frame student rollout, ONLINE teacher
# training, NO schedule. Based on baseline_G_NR (best phase-3
# configuration to date).
#
# Training plan:
#   * Steps [0, 10):   Fresh rank-32 LoRA FROZEN. DMD scores against
#                      the (v14-merged + zero-output rank-32) teacher.
#                      No aux loss, no LoRA gradient, no LoRA optim step.
#   * Steps [10, 100): Fresh rank-32 LoRA TRAINING ONLINE with linearly
#                      ramping aux loss weight (0 → 1.0). Every gen
#                      step runs the aux pass: LoRA forward on noisy
#                      student-gen latents → flow loss vs GT →
#                      backward → LoRA optim step. DMD continues to
#                      score against the (now evolving) LoRA each iter.
#   * Steps [100, 200): LoRA training at full aux loss weight.
#
# LoRA setup: v14's rank-256 LoRA is MERGED INTO THE BASE at startup
# (v14's learned behavior folded permanently into the WAN base
# weights), then a FRESH rank-32 LoRA is applied on top. The fresh
# LoRA starts from zero output and is what the aux teacher trains.
# Trade-off vs the prior "keep v14 rank-256 LoRA trainable" design:
# +1 GB memory savings (smaller adapter + Adam state + grads), at the
# cost of restarting LoRA learning from zero rather than fine-tuning
# v14's already-trained adapter. The merged-v14 base preserves v14's
# behavior, so DMD scoring remains anchored to v14's distribution.
#
# Differences vs baseline_G_NR:
#
#   1. real_teacher_train_online = true (was false). v14 is merged
#      into the base, then a fresh rank-32 LoRA
#      (``aux_teacher_lora_rank=32``, ``aux_teacher_lora_alpha=32``)
#      is applied on top; the fresh LoRA receives flow-loss gradient
#      against GT every iter from step 10 onward via the aux pass.
#
#   2. aux_teacher_loss_weight = 1.0 (was 0.0). Active aux loss.
#
#   3. real_teacher_input_source = student. LoRA's noisy_x is built
#      from the student-gen rollout EVERY iter — no GT mixing, no
#      Bernoulli coin, no schedule. LoRA learns "given a noisy version
#      of the student's prediction, point to GT." Pure student-
#      distribution input throughout the training window.
#
#   4. aux_teacher_p_schedule_enabled = false. No curriculum. Betting
#      the LoRA can converge on the student distribution from step
#      30 onward without a GT-warmup phase. The scheduled variant
#      (curriculum 1.0 → 0.0 over 100 steps) lives in
#      ``train_action_forcing_phase3_online_teacher_G_NR_schedule.sbatch``
#      for reference; we're not running it for now.
#
#   5. aux_teacher_send_student_grad = false. Implicit gradient
#      channel back to the student via noise_base = chunk is severed.
#      LoRA trains in isolation; DMD remains the sole student-grad
#      path through real_score. Clean gradient routing — the student
#      is not double-pushed by aux + DMD.
#
#   6. aux_teacher_start_step = 10 + aux_teacher_loss_warmup_steps = 90.
#      Combined hard gate + linear loss-weight ramp on the aux pass:
#        Steps [0, 10):   LoRA frozen — zero aux loss, no real_score
#                         forward, no LoRA gradient, no LoRA optim step.
#        Steps [10, 100): LoRA training with linearly ramping aux loss
#                         weight: 0 at step 10 → ``aux_teacher_loss_
#                         weight`` (1.0) at step 100. Effective weight
#                         per step = 1.0 * (step - 10) / 90.
#        Steps [100, 200): LoRA training at full aux loss weight (1.0).
#      The clean_x for the aux pass is a STABLE reference: GT
#      last-seed-chunk + the first 6 student chunks' post-Step-3.3.5
#      refined cache_pred (snapshotted on iter 1 from the pipeline's
#      ``_clean_chunk`` buffer; reset on streaming reset). Detached.
#
#   7. real_teacher_warmup_steps = 0 (was 200 in v25). The start-step
#      gate + loss ramp is the only delay; no LR warmup. Once aux
#      training starts at step 10, LR is immediately at base value
#      (5e-5).
#
# Everything else (max_steps=200, wandb_project=longlive-phase-3,
# GAN+SAM2+MANIQA wiring, frame counts, RoPE alignment) inherits
# unchanged from baseline_G_NR.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_G_NR
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
echo "Starting Phase-3 online-teacher G_NR (no schedule) on $(date)"
echo "  * 21-frame student rollout, 200-step max."
echo "  * Teacher: v14 LoRA, ONLINE training starting at step 30."
echo "  * Steps [0, 30): LoRA frozen. Steps [30, 200): LoRA trains on"
echo "    student-gen rollouts to predict GT-equivalent (no schedule)."
echo "  * aux_teacher_send_student_grad=false (DMD is sole student-grad path)."
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
    max_rolls_per_ride=300 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=21 \
    num_workers=0 \
    boundary_vae_roundtrip=true \
    local_attn_size_schedule=[[0,21]] \
    dmd_context=self \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=1.0 \
    real_teacher_lr=5e-5 \
    real_teacher_warmup_steps=0 \
    real_teacher_input_source=student \
    aux_teacher_p_schedule_enabled=false \
    aux_teacher_send_student_grad=false \
    aux_teacher_start_step=20 \
    aux_teacher_loss_weight=1.0 \
    aux_teacher_loss_warmup_steps=180 \
    aux_teacher_lora_rank=32 \
    aux_teacher_lora_alpha=32 \
    dmd_loss_start_step=50 \
    dmd_loss_warmup_steps=50 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    gan_loss_weight=0.25 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=768 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=50 \
    gan_warmup_steps=150 \
    gan_sam2_frame_pool=none \
    gan_sam2_encoder_chunk_size=84 \
    maniqa_approx_loss_weight=1.0 \
    maniqa_n_frames=84 \
    maniqa_n_patches_per_frame=8 \
    maniqa_metric_name=maniqa-pipal \
    gan_critic_dense_distillation=true \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=action_forcing_phase3_online_teacher_G_NR_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
