#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-G-NR-v6-no-streaming
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
# Phase-3 online_teacher_G_NR_v6_no_streaming: rebalances the head
# weights so DMD becomes the dominant student-side signal, with GAN
# and aux teacher pulled back to support roles. Builds on v5 (teacher_
# cadence='fake' inner loop, decaying GT-mix on DMD's clean_x).
#
# DESIGN INTENT — WHY THIS LOSS COMBINATION
# -----------------------------------------
# The whole recipe is structured around one observation about DMD:
# it's an exceptional signal for 3D world understanding — the
# pred_real - pred_fake gradient inherits the teacher's
# distribution-matching pressure, which gives the student real
# structural and temporal grounding that no perceptual or
# adversarial head can provide alone. That's why DMD is the primary
# pull on the student and why it's active from step 1. But DMD also
# has a known failure mode: a positive feedback loop where the
# student starts making the *teacher's* failure modes, and the
# teacher reinforces them. Once that loop closes, the loss looks
# fine even as the output collapses.
#
# Each of the other heads exists to absorb a specific failure mode
# of the primary DMD signal:
#
#   MANIQA — the "is this a real image?" anchor.
#     Catches the teacher's failure modes by penalising outputs that
#     look obviously non-image-manifold. Active from step 0 because
#     it has to be in place BEFORE the student starts drifting.
#     MANIQA-alone is also a known failure mode: the student forgets
#     it's modelling a 3D scene at all and produces image-manifold
#     soup, eventually collapsing to gray. So MANIQA isn't a
#     standalone trainer — it's an "image manifold" guard for the
#     other signals.
#
#   MANIQA + DMD (no GAN) — does well for a while, then collapses
#     around step 61 in this codebase. The mechanism we observe:
#     one or two bad student rollouts in a row destabilise the
#     teacher (which is being trained on student outputs), the
#     teacher then puts out pred_real that pulls the student further
#     off, and the perceptual regularisation isn't enough structural
#     pressure to hold the student on the GT manifold. The collapse
#     isn't gradual — it's a series of stable states that converge
#     to instability fast.
#
#   GAN — the "is this a coherent 3D scene?" anchor.
#     The R3GAN + SAM2 distilled critic puts pressure on the
#     structural and spatial composition of the output, which is
#     exactly the signal MANIQA can't provide. Sweet-spot is
#     narrow: too much GAN → cartooniness (D becomes the
#     attractor); way too much → discriminator collapse and full
#     destabilisation (observed in baseline_G_NR v1 at effective
#     weight ≈0.51). Just enough → actively prevents the MANIQA+DMD
#     collapse, doesn't merely delay it. v6's gan_loss_weight=0.05
#     plateau sits in the lower part of that sweet spot, well clear
#     of the empirical 0.40 stable ceiling.
#
# So the three student-side losses are deliberately complementary:
# DMD provides the 3D-world grounding, MANIQA guards the image
# manifold, GAN guards the scene structure. Each one alone is a
# known failure mode; the combination has — in this codebase up
# until now — been the only configuration that delays collapse
# meaningfully but still hasn't actually prevented it.
#
# WHY DMD KEEPS WINNING ANYWAY (THE REAL PROBLEM)
# -----------------------------------------------
# DMD loss is COMPOUNDING. Every step it nudges the student
# distribution toward the teacher's, and over enough steps the
# student drifts to the teacher's distribution — NOT GT. The
# teacher was supposed to capture GT, but in practice it can't:
#   * it's being fine-tuned online so its own targets are moving,
#   * it can't fully encode the causal/autoregressive setup the
#     student operates in (the teacher is bidirectional during DMD
#     scoring; the student is causal at inference time),
#   * and a freshly-initialised rank-32 LoRA isn't a perfect bridge
#     for the (clean_x = student-cache | noisy = student-rollout,
#     bidirectional) distribution shift.
#
# Evidence: the per-key MAE-vs-GT trace shows huge spikes right
# after the student collapses, even though gen/dmd_loss stays
# roughly the same magnitude. The student is happily "matching the
# teacher" while moving away from GT. DMD's compounding nature
# converges on instability through a series of stable-looking
# intermediate states — exactly the behaviour we've been observing.
#
# REGULARISERS DELAY COLLAPSE; ONLINE TEACHER IS WHAT FIXES IT
# ------------------------------------------------------------
# MANIQA and GAN can slow this drift but can't reverse it — they
# don't see GT directly, only the student's outputs. To actually
# anchor the student to GT, the teacher's own predictions have to
# track GT, which is what the online aux pass does: the LoRA is
# trained on flow-loss against GT every iter, and its (= the
# teacher's) updated predictions then flow back into DMD scoring.
# That's the fundamental fix.
#
# The trap we keep hitting: we've previously wired the aux teacher
# too strongly. The symptom is "even during and after collapse,
# MAE-vs-GT doesn't move all that much" — the student isn't drifting
# *because* of distribution-matching anymore, it's actively
# MEAN-SEEKING because the LoRA's GT-trained signal is so strong
# that DMD effectively becomes a teacher-mediated MSE-to-GT loss.
# That's a known DMD failure mode and the earliest discussion in
# this thread flagged it; the experimental results have now backed
# it up.
#
# So v6's plan is: keep DMD as the primary push, keep MANIQA + GAN
# as the image-manifold and structural anchors against the teacher's
# failure modes, and progressively WEAKEN the aux teacher's
# mean-seeking pull (aux_teacher_loss_weight cut 1.0 → 0.35 in v6,
# expected further reductions in subsequent versions). The
# expectation is that there's a sweet spot where the aux teacher
# corrects the teacher's drift enough that DMD doesn't compound to
# collapse, but not so much that DMD becomes a covert MSE-to-GT
# loss. v6 is a step toward finding that sweet spot.
#
# SCHEDULE-SEQUENCING RATIONALE
# -----------------------------
# MANIQA's approxes (the small distilled nets that stand in for the
# heavy MANIQA-pipal model in the autograd graph) kick into active
# training at step 20 and stabilise by step 30 — observed empirically
# as the sharp "additive bias" drop on the GAN-loss-weighted trace
# between steps 21 and 31, which is the perceptual-loss component
# converging onto sensible target magnitudes. During that bootstrap
# window MANIQA's gradient magnitudes are large and noisy, which
# destabilises the GAN if its loss path is "cinched" at the same
# time. v6 sequences the heads so the GAN gen-side doesn't engage
# until step 30 — by then MANIQA has anchored the student onto the
# real image manifold and the GAN can start adding structural
# pressure to outputs that are already roughly in-distribution.
# Aux teacher is also started at step 30 for the same reason: it
# shares chunks with MANIQA's gradient flow, and a freshly-
# initialised rank-32 LoRA bootstrapped on top of MANIQA's
# transient would compound noise. DMD is the exception — it starts
# at step 1 because its scoring math doesn't depend on any learned
# approximator, so its memory + numerical profile is stable from
# iter 1 onward.
#
# v6 DELTAS vs v5
# ---------------
#
#   1. DMD now active from step 1 (was step 50), weight cap 0.5
#      (was implicit 1.0). The DMD ramp still spans 100 steps so its
#      effective weight grows linearly to 0.5 by step 101. Memory
#      cost: DMD forward+backward in the autograd graph from iter 1.
#
#      step  1 → effective DMD weight = 0
#      step 51 → effective = 0.5 * 50/100 = 0.25
#      step 101 → effective = 0.5 (plateau)
#
#      Start step is 1 (NOT 0) so the very first iter has no DMD
#      activations live — gives the allocator one clean iter to
#      consolidate before DMD activations join the gen-backward's
#      retained graph (the GPU is tight under the full
#      DMD+GAN+MANIQA+LoRA-aux+aux-Inner-loop stack on iter 0).
#
#   2. GAN gen-side cap cut from 0.25 → 0.05 (5x weaker), with
#      gan_critic_warmup_steps=30 (was 10) and gan_warmup_steps=100
#      (was 200). With critic_warmup=30 the discriminator trains
#      alone for the first 30 steps, then the gen-side ramps 0 → 0.05
#      over the following 100 steps, reaching the 0.05 plateau at
#      step 130. Terminal effective weight ≈0.05 — well below the
#      0.40 empirical stable ceiling (lots of margin). critic_warmup
#      bumped to 30 specifically so the GAN gen-side doesn't engage
#      until after the MANIQA approx has stabilised (see SCHEDULE-
#      SEQUENCING RATIONALE above).
#
#   3. Aux teacher loss weight cap cut from 1.0 → 0.35 (3x weaker),
#      aux_teacher_start_step bumped 20 → 30, aux_teacher_loss_warmup
#      _steps trimmed 80 → 70. New aux schedule: frozen [0, 30),
#      ramps 0 → 0.35 over [30, 100), plateaus from step 100. The
#      LoRA pointing-vehicle reaches full strength at step 100, 1
#      step before DMD's effective weight crosses 0.25. The +10 bump
#      on start_step (20 → 30) again sequences this head to engage
#      only after MANIQA's approx has settled — the LoRA's training
#      shares chunks with the same gen iter where MANIQA's gradient
#      flows, so an unstable MANIQA pass during LoRA bootstrap would
#      compound the noise on a freshly-initialised rank-32 LoRA.
#
#   4. real_teacher_warmup_steps doubled 50 → 100 (LR warmup on the
#      LoRA's AdamW). At step 30 (first aux fire) the LoRA's LR is
#      now 31% of base (vs 62% in v5), reaching full LR at step 100
#      coinciding with the loss-weight plateau.
#
# All other v5 knobs preserved:
#   * teacher_cadence='fake' — inner loop fires dfake_gen_update_ratio
#     (=5) LoRA optimizer steps per outer iter once aux is open.
#   * dmd_context='mix' + dmd_context_mix_p schedule 0.30 → 0.00 over
#     100 steps — DMD-scoring clean_x has a brief GT anchor that
#     decays as the LoRA pointing-vehicle settles.
#   * real_teacher_input_source='student', causal_mask=false.
#   * max_rolls_per_ride=1 (no-streaming, fresh ride per step).
#   * dmd_frozen_teacher_pass_enabled=false (LoRA is sole pointing
#     vehicle).
#   * aux_teacher_send_student_grad=false (two-stream gradient
#     separation preserved).
#
# WHAT THIS CHANGES IN STUDENT-SIDE PUSH ATTRIBUTION
# --------------------------------------------------
# At step 150 (everything plateaued / nearly so) the student
# gradient is approximately a sum of:
#   * DMD term — weight 0.5 (full)
#   * GAN gen-side — weight 0.05 (full, but 10x weaker than DMD)
#   * MANIQA approx perceptual loss — weight 1.0 (full from step 0)
#
# v6 deliberately makes the MANIQA bucket the largest static weight
# but expects DMD to dominate the actual *gradient magnitude* on the
# student because DMD's raw scale (pred_real - pred_fake) is large
# relative to the perceptual approx's typical loss values. Confirm
# this empirically via the per-key contribution traces on wandb.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_G_NR_v6_no_streaming
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
echo "Starting Phase-3 online-teacher G_NR v6 (no streaming) on $(date)"
echo "  * 21-frame iter-1-only student rollout, 200-step max."
echo "  * max_rolls_per_ride=1 — streaming resets every step (no iter k>=2)."
echo "  * Teacher: fresh rank-32 LoRA on v14-merged base (alpha=64, r=32)."
echo "  * teacher_cadence=fake — LoRA optim fires 5x per outer iter."
echo "  * Steps [0, 30): LoRA frozen. Steps [30, 100): aux loss weight ramps"
echo "    0 -> 0.35. Steps [100, 200): aux at 0.35 plateau."
echo "  * GT-mix in DMD clean_x: 0.30 -> 0.00 linearly over first 100 steps."
echo "  * DMD cap=0.5, dmd_loss_start_step=1, dmd_loss_warmup_steps=100"
echo "    (full at step 101 — DMD is the primary student-push)."
echo "  * GAN cap=0.05, gan_critic_warmup=30, gan_warmup=100 (full at"
echo "    step 130). Much weaker than v5 (was 0.25 cap)."
echo "  * Aux teacher cap=0.35 (was 1.0 in v5) — supporting role."
echo "  * real_teacher_warmup_steps=100 (was 50) — slower LoRA LR ramp."
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
    real_teacher_warmup_steps=100 \
    real_teacher_input_source=student \
    teacher_cadence=fake \
    aux_teacher_p_schedule_enabled=false \
    aux_teacher_send_student_grad=false \
    aux_teacher_start_step=30 \
    aux_teacher_loss_weight=0.35 \
    aux_teacher_loss_warmup_steps=70 \
    aux_teacher_lora_rank=32 \
    aux_teacher_lora_alpha=64 \
    dmd_loss_weight=0.5 \
    dmd_loss_start_step=1 \
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
    gan_critic_warmup_steps=30 \
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
    run_name=action_forcing_phase3_online_teacher_G_NR_v6_no_streaming_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
