#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-G-NR-v3-no-streaming
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
# Phase-3 online_teacher_G_NR_v3_no_streaming: same plumbing as v2
# (online teacher, no streaming, MANIQA from step 0, aux LoRA from
# step 20) with the GAN, DMD, and aux ramps retuned so the LoRA
# pointing-vehicle is meaningfully trained by the time DMD relies on
# it, and with extra GAN-side stability margin pulled from the
# baseline_G_NR v1 trace.
#
# LORA-AS-POINTING-VEHICLE DESIGN INTENT
# --------------------------------------
# v14 was trained CAUSALLY with teacher-forced clean GT context
# (clean_GT → noised_GT). The phase-3 setup feeds the teacher
# BIDIRECTIONAL attention + clean_x = [GT-seed | student-refined-
# cache_pred] + noised-student-rollout — a distribution shift on
# every axis (mask, context, input). The fresh rank-32 LoRA exists
# to bridge that shift: it learns to map noised-student-distribution
# inputs to GT under the new attention/context regime.
#
# Knobs aligned with this intent:
#   * real_teacher_input_source=student — LoRA capacity goes entirely
#     into the bridge function it actually needs to learn. Mixing in
#     GT inputs would dilute the budget on a task that's not what
#     the LoRA gets used for (DMD always scores student-rollouts).
#   * real_teacher_causal_mask=false — bidirectional, matching how
#     the teacher will be queried at scoring time.
#   * dmd_frozen_teacher_pass_enabled=false — LoRA is the SOLE
#     pointing-vehicle. No GT-anchored second branch.
#   * aux_teacher_send_student_grad=false — two independent gradient
#     flows. LoRA is severed from the student so the teacher never
#     becomes a co-adaptation partner.
#
# GAN-side rationale (from baseline_G_NR v1 trace, slurm 4513839):
#   * v1 (loss_weight=1.0, warmup=80, critic_warmup=10) destabilised
#     around step 46. The SAM2 distilled critic collapsed onto D
#     (c_corr → +1.0), D lost real/fake separation, gen_grad jumped
#     10×, gen_loss went negative.
#   * Empirical stable ceiling for the effective G-side weight is
#     ~0.40; collapse-onset sits at ~0.51.
#   * Here the cap is 0.25 with the warmup stretched across the full
#     A/B horizon — terminal effective weight 0.2375, well below the
#     0.40 stable ceiling. Large safety margin since this run also
#     carries an online-trained LoRA on the critical path and we want
#     headroom against any GAN/LoRA interaction.
#   * Generator is pretrained, so the critic can come up quickly
#     (critic_warmup=10) without giving D a head start the generator
#     can't match.
#
# NO-STREAMING MECHANISM (unchanged from v2)
# ------------------------------------------
# max_rolls_per_ride = 1. Each training step starts a fresh ride
# (iter 1) and resets the streaming state immediately after the
# single chunk is processed. Iter 1 forces ``new_frames =
# chunk_size = 21`` (the full 7-block rollout); the variable-frame
# rollout that iter k≥2 would sample from {3, 6, 9, 12, 15, 18, 21}
# never fires. ~7× more rides loaded from disk per training run.
#
# Training plan (steps 0..200):
#   * Steps [0, 20):   Fresh rank-32 LoRA FROZEN. DMD is off
#                      (dmd_loss_start_step=50). No aux loss, no LoRA
#                      gradient, no LoRA optim step.
#   * Steps [20, 100): Fresh rank-32 LoRA TRAINING ONLINE with
#                      linearly ramping aux loss weight (0 → 1.0 over
#                      80 steps). Effective weight per step =
#                      1.0 * (step - 20) / 80. Plateau reached at
#                      step 100 — 50 steps before DMD itself hits
#                      full strength (step 150), so the pointing-
#                      vehicle is fully active before DMD's full
#                      reliance kicks in.
#   * Steps [100, 200): Aux LoRA at full loss weight throughout. By
#                      step 150 the LoRA has had 50 steps of plateau-
#                      rate training behind it before DMD reaches
#                      full strength.
#
# WHAT INFORMS THE STUDENT vs WHAT INFORMS THE LoRA
# -------------------------------------------------
# Two independent gradient flows: the student is updated by the gen
# optim step (driven by DMD + GAN + MANIQA, all weighted), and the
# LoRA is updated by its own aux optim step (driven by flow-loss
# against GT). The aux pass is severed from the student
# (aux_teacher_send_student_grad = false), so LoRA training does
# NOT feed back into the student.
#
# Student-bound losses, by step window (v3 schedule):
#   Steps [0, 10):
#     - MANIQA (NR-IQA, weight 1.0 from step 0, no warmup) — ONLY
#       student-bound signal in this window.
#     - GAN gen-side: 0 (critic trains alone for 10 steps).
#     - DMD: 0 (dmd_loss_start_step=50).
#     - Aux: severed from student.
#   Steps [10, 50):
#     - MANIQA at full weight.
#     - GAN gen-side starts ramping: effective weight =
#       0.25 * (step - 10) / 200. At step 50 effective = 0.05.
#     - DMD: still 0.
#   Steps [50, 150):
#     - MANIQA at full weight.
#     - DMD ramps 0 → 1 over 100 steps (dmd_loss_warmup_steps=100).
#       At step 100 DMD is at 0.50; at step 150 DMD is full.
#     - GAN gen-side continues ramping. At step 150 effective =
#       0.25 * 140/200 = 0.175.
#   Steps [150, 200):
#     - MANIQA + DMD at full weight; LoRA pointing-vehicle plateaued
#       since step 100 — this is the "everything-on" regime that
#       actually tests the design.
#     - GAN gen-side continues ramping monotonically. Terminal
#       weight at step 200 = 0.25 * 190/200 = 0.2375 — ~59% of the
#       0.40 empirical stable ceiling, only reached at run end.
#
# LoRA training progress vs DMD reliance:
#   step  20 → aux fires; loss weight 0,    DMD weight 0
#   step  50 → loss weight 0.375,           DMD just turning on
#   step 100 → loss weight 1.0 (plateau!), DMD at 0.50
#   step 150 → loss weight 1.0,             DMD full
#   step 200 → loss weight 1.0,             DMD full (50 plateau steps)
#
# Differences vs v2:
#
#   1. gan_critic_warmup_steps = 10 (was 35). Generator is pretrained,
#      so the SAM2 critic can come up fast without giving D a long
#      monopoly over the GAN pass. Matches the baseline_G_NR v1
#      setting that didn't show critic-side problems — what failed in
#      v1 was the gen-side weight, not the critic warmup.
#
#   2. gan_warmup_steps = 200 (was 150). Ramp now spans the full A/B
#      horizon — pure ramp regime, no plateau training inside the
#      window. Effective gen-side GAN weight at any step is
#      0.25 * min(1, max(0, (step - 10) / 200)), reaching 0.2375 at
#      step 200.
#
#   3. aux_teacher_loss_warmup_steps = 80 (was 180). Aux loss weight
#      reaches full at step 100 instead of step 200, so the LoRA gets
#      ~100 steps of plateau-rate training inside the A/B horizon
#      (vs zero in v2). The pointing-vehicle is actually pointing
#      before DMD heavily relies on it.
#
#   4. aux_teacher_lora_alpha = 64 (was 32). α/r = 2.0 (was 1.0) —
#      standard LoRA recipe scaling. Doubles the effective update
#      rate of the LoRA without changing optimizer LR, since the
#      LoRA is doing real distribution-shift work (bidirectional +
#      new clean_x context + student inputs).
#
#   5. dmd_loss_start_step = 50 (was 55). 5 steps earlier so DMD
#      onset aligns with the start of GAN gen-side ramp arriving at a
#      meaningful level (~0.05).
#
#   6. dmd_loss_warmup_steps = 100 (was 50). Slower DMD ramp,
#      doubled. Reaches full at step 150 instead of step 105. Lets
#      the LoRA pointing-vehicle settle into plateau (at step 100)
#      before DMD's full weight engages, and keeps the DMD/GAN/LoRA
#      ramps from all crossing their high-derivative regions at the
#      same time.
#
# All other v2 settings preserved: aux_teacher_start_step=20,
# real_teacher_lr=5e-5, real_teacher_warmup_steps=50,
# real_teacher_input_source=student, max_rolls_per_ride=1, online
# teacher wiring, send_student_grad=false routing.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_G_NR_v3_no_streaming
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
echo "Starting Phase-3 online-teacher G_NR v3 (no streaming) on $(date)"
echo "  * 21-frame iter-1-only student rollout, 200-step max."
echo "  * max_rolls_per_ride=1 — streaming resets every step (no iter k>=2)."
echo "  * Teacher: fresh rank-32 LoRA on v14-merged base (alpha=64, r=32)."
echo "  * Steps [0, 20): LoRA frozen. Steps [20, 100): aux loss weight ramps"
echo "    0 -> 1.0. Steps [100, 200): LoRA pointing-vehicle at plateau."
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
    dmd_context=self \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=1.0 \
    real_teacher_lr=5e-5 \
    real_teacher_warmup_steps=50 \
    real_teacher_input_source=student \
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
    gan_loss_weight=0.1 \
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
    run_name=action_forcing_phase3_online_teacher_G_NR_v3_no_streaming_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
