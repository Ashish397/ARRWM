#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-NR-v2-no-streaming
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
# Phase-3 online_teacher_NR_v2_no_streaming: 21-frame student
# rollout, ONLINE teacher training, NO streaming (iter-1-only per
# ride), GAN FULLY DISABLED, with slower aux ramp tuned for stability.
# Iter k≥2 is deferred to phase 4. Based on v1
# (train_action_forcing_phase3_online_teacher_G_NR.sbatch) with the
# knob deltas listed under "Differences vs v1" below.
#
# Variant identity: this is the "NR" (no R3GAN) sibling of
# online_teacher_G_NR_v2_no_streaming. Every other knob matches; the
# only delta is gan_enabled=false (and therefore zero GAN compute).
#
# NO-STREAMING MECHANISM
# ----------------------
# max_rolls_per_ride = 1 (was 300). Each training step starts a fresh
# ride (iter 1) and resets the streaming state immediately after the
# single chunk is processed. Iter 1 forces ``new_frames =
# chunk_size = 21`` (the full 7-block rollout); the variable-frame
# rollout that iter k≥2 would sample from {3, 6, 9, 12, 15, 18, 21}
# never fires. The streaming machinery (setup_sequence, KV cache,
# clean_chunk buffer, _clean_chunk snapshot for aux clean_x) still
# runs unchanged — we just never re-enter it within a ride. This is
# a strict subset of the v1 training distribution: same iter-1 path,
# just no iter-2+ continuation. Cost: more rides loaded from disk per
# training run (we throw away the warm KV cache every step instead of
# reusing it across chunks); the exact multiplier depends on the ride
# distribution. Acceptable trade for stability.
#
# GAN DISABLED
# ------------
# gan_enabled = false. This is the hard master-switch — when false:
#   * The R3GAN discriminator is NOT constructed (no SAM2 encoder
#     load, no critic params allocated, no critic optimizer).
#   * No gen-side GAN loss is added to the student.
#   * No critic-side adversarial pass runs each step.
# This is materially different from gan_loss_weight=0.0 with
# gan_enabled=true (which keeps the critic training but contributes
# zero gradient to the student) — that variant still pays SAM2 forward
# + critic backward every step. Here both are skipped entirely.
# The gan_* knobs in the override below (gan_loss_weight,
# gan_*_warmup, gan_sam2_*, gan_critic_*) are INERT under
# gan_enabled=false. They are retained verbatim from the G_NR sibling
# so a side-by-side diff between the two sbatches is one-line clean.
# The SAM2 path sanity checks at the top of the script also remain;
# they are cheap and protect against silently re-enabling GAN later
# without the checkpoint installed.
#
# Training plan (steps 0..200):
#   * Steps [0, 20):   Fresh rank-32 LoRA FROZEN. DMD scores against
#                      the (v14-merged + zero-output rank-32) teacher.
#                      No aux loss, no LoRA gradient, no LoRA optim step.
#   * Steps [20, 200): Fresh rank-32 LoRA TRAINING ONLINE with linearly
#                      ramping aux loss weight (0 → 1.0 over 180
#                      steps). Effective weight per step =
#                      1.0 * (step - 20) / 180. The ramp reaches full
#                      weight exactly at step 200 (run end).
#
# WHAT INFORMS THE STUDENT vs WHAT INFORMS THE LoRA
# -------------------------------------------------
# Two independent gradient flows: the student is updated by the gen
# optim step (driven by DMD + MANIQA, both weighted; GAN disabled),
# and the LoRA is updated by its own aux optim step (driven by flow-
# loss against GT). The aux pass is severed from the student
# (aux_teacher_send_student_grad = false), so LoRA training does
# NOT feed back into the student.
#
# Student-bound losses, by step window (NR v2 schedule):
#   Steps [0, 55):
#     - MANIQA (NR-IQA, weight 1.0 from step 0, no warmup) — ONLY
#       student-bound signal in this window.
#     - DMD: weight = 0 (dmd_loss_start_step=55).
#     - GAN: DISABLED — zero contribution at every step.
#     - Aux: severed from student.
#   Steps [55, 105):
#     - MANIQA at full weight.
#     - DMD ramps 0 → 1 over [55, 105) (dmd_loss_warmup_steps=50).
#     - GAN: DISABLED.
#   Steps [105, 200):
#     - MANIQA + DMD both at full weight.
#     - GAN: DISABLED.
#
# Differences vs v1 (online_teacher_G_NR.sbatch):
#
#   1. max_rolls_per_ride = 1 (was 300). NO STREAMING — iter 1 only;
#      streaming resets every step. See "NO-STREAMING MECHANISM" above.
#
#   2. gan_enabled = false (was true). GAN FULLY DISABLED — no SAM2
#      encoder, no discriminator, no critic optim, no gen-side GAN
#      loss. Saves the GAN head's memory + compute every step. See
#      "GAN DISABLED" above. This is the defining difference between
#      this NR sbatch and its G_NR sibling.
#
#   3. aux_teacher_start_step = 20 (was 10). LoRA stays frozen 10
#      steps longer so the student has more time to anneal the
#      MANIQA-only signal before the aux LoRA starts moving.
#
#   4. aux_teacher_loss_warmup_steps = 180 (was 90). Aux loss weight
#      ramps over 180 steps from step 20 to step 200, reaching 1.0
#      exactly at run end. Slower bring-up than v1, which hit full
#      weight at step 100.
#
#   5. dmd_loss_start_step = 55 (was 50). DMD stays off for the first
#      55 steps, then ramps 0 → 1 over dmd_loss_warmup_steps=50,
#      reaching full weight at step 105. Slight delay vs v1 to keep
#      the schedule aligned with the G_NR sibling (matching cadence
#      makes head-to-head comparison cleaner).
#
# Inert knobs (gan_loss_weight=0.0, gan_critic_warmup_steps=35,
# gan_warmup_steps=150, gan_sam2_*, gan_critic_*): retained in the
# override for diff parity with the G_NR sibling. Under
# gan_enabled=false these have NO effect on training.
#
# LoRA setup, aux pass plumbing, RoPE alignment, DMD-vs-LoRA scoring,
# clean_x assembly, send_student_grad=false routing: all inherited
# unchanged from v1.
#
# Everything else (max_steps=200, wandb_project=longlive-phase-3,
# MANIQA wiring, frame counts) inherits from v1.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_NR_v2_no_streaming
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
echo "Starting Phase-3 online-teacher NR v2 (no streaming, GAN disabled) on $(date)"
echo "  * 21-frame iter-1-only student rollout, 200-step max."
echo "  * max_rolls_per_ride=1 — streaming resets every step (no iter k>=2)."
echo "  * GAN DISABLED (gan_enabled=false): no SAM2, no critic, no GAN loss."
echo "  * Teacher: fresh rank-32 LoRA on v14-merged base."
echo "  * Steps [0, 20): LoRA frozen. Steps [20, 200): LoRA aux ramps 0 -> 1.0."
echo "  * Student first-55-steps signal: MANIQA only (DMD warming up)."
echo "  * dmd_loss_start_step=55, dmd_loss_warmup_steps=50 (full at step 105)."
echo "  * aux_teacher_send_student_grad=false (DMD is sole student-grad path"
echo "    from the teacher; aux LoRA optim is severed from the student)."
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
    dmd_loss_start_step=55 \
    dmd_loss_warmup_steps=50 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    gan_enabled=false \
    gan_loss_weight=0.0 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=768 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=35 \
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
    run_name=action_forcing_phase3_online_teacher_NR_v2_no_streaming_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
