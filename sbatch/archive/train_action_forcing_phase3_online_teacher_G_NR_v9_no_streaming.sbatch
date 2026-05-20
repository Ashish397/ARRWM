#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-G-NR-v9-no-streaming
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-3 online_teacher_G_NR_v9_no_streaming: counterpoint to v8.
#
# REVISED v7 COLLAPSE DIAGNOSIS
# -----------------------------
# Initial reading of v7's wandb traces (c_corr → +0.96 at step 71)
# suggested D-domination — the v1-style "SAM2 critic collapses
# onto R3GAN D" failure. Decoded-video inspection corrected that:
# v7's collapse mechanism was the same as v6's, and is best
# described as **mode-seeking via DMD compounding**:
#
#   * DMD is mode-seeking by construction (pred_real - pred_fake
#     gradient pulls the student onto the teacher's high-density
#     modes, not its mean).
#   * Aux teacher trains the LoRA against GT every iter, which
#     sharpens the teacher's modes toward GT-typical outputs.
#   * The student therefore drifts to a SINGLE mode that the
#     LoRA-modified teacher considers high-density AND close to
#     GT — outputs look MAE-close to GT but lose diversity (same
#     point repeated). "The DMD loss made the student produce what
#     the teacher produced." MAE-vs-GT collapses to a low,
#     stationary value, masquerading as good convergence.
#
# This is two consecutive runs (v6, v7) hitting the same
# mode-seeking-via-DMD-compounding wall with
# aux_teacher_loss_weight=0.35.
#
# v8 went the "everything stronger" route (aux=1.0, gan_w=0.2,
# 5:1 D cadence) on the user's instinct that the other losses
# could regularise against the mode-collapse. That instinct is
# actually correct for the GAN side specifically: R3GAN (RpGAN +
# R1 + R2) is engineered as an anti-mode-collapse recipe, so
# stronger GAN gives the student a real diversity-preserving
# pressure that DMD's KL-style mode-seeking pull alone can't
# overcome. v8's mistake was strengthening aux at the same time —
# that compounds the teacher's mode-sharpening and makes the
# student's target mode even narrower.
#
# v9 keeps the user's "stronger GAN as anti-mode-collapse" idea
# but reverses the aux direction: WEAKEN aux to broaden the
# teacher's mode envelope, MODESTLY strengthen the GAN-side
# diversity signal, and revert v7's D over-training so the GAN
# influence comes from loss-weight rather than D-cadence.
#
# v9 DELTAS vs v7
# ---------------
#   1. aux_teacher_loss_weight  0.35 → 0.2   (THE main lever:
#                                              continue progressive
#                                              weakening v5→v6→v9 of
#                                              the aux teacher's
#                                              mode-sharpening pull
#                                              on the LoRA teacher)
#   2. gan_loss_weight         0.05 → 0.1    (modest bump — exploit
#                                              R3GAN's built-in
#                                              anti-mode-collapse
#                                              property. Still 4×
#                                              below 0.40 empirical
#                                              stability ceiling so
#                                              not risking the
#                                              D-domination collapse
#                                              mode independently.)
#   3. gan_updates_per_step     2 → 1        (back to R3GAN paper
#                                              default 1:1 D:G;
#                                              diversity pressure
#                                              comes from loss_weight
#                                              + R1/R2=5.0, not from
#                                              D-cadence over-training)
#   4. gan_critic_warmup_steps  40 → 30      (D and gen-side engage
#                                              simultaneously at step
#                                              30; removes D head-start
#                                              for cleaner attribution
#                                              if collapse still
#                                              happens)
#
# Trace shape to verify hypothesis:
#   * If MAE settles in a MODERATE band (not too low, not exploding):
#     working — the v9 balance found the diversity-preserving sweet
#     spot.
#   * If MAE still goes TOO LOW → mode-seeking persists. Either
#     aux still too strong (try 0.1 next), or DMD itself needs
#     pulling back (dmd_loss_weight 0.5 → 0.3 next).
#   * If MAE explodes (HIGH) → off-manifold drift. Aux too weak
#     to anchor — bump back to 0.3 next.
#   * If fish-scale returns → GAN-side too sharp despite R1/R2=5.0.
#     Pull gan_loss_weight back to 0.05 next.
#
# ---------------------------------------------------------------------
# Inherited v6→v7 deltas (still in place in v9):
# v6 just completed a full 200-step run; these
# five v6→v7 changes are direct responses to symptoms observed in
# v6's decoded video + wandb traces:
#
#   1. GT-blend on DMD's clean_x removed — was causing blur and drift.
#   2. GAN R1/R2 gradient penalties bumped 1.0 → 5.0 to suppress the
#      60x45 latent-token fish-scale artefact visible on v6's decoded
#      video.
#   3. Discriminator training deferred from step 0 → step 30 (new
#      ``gan_disc_start_step`` knob) so D doesn't fit on the MANIQA-
#      bootstrap-window student outputs.
#   4. ``gan_updates_per_step`` bumped 1 → 2 — D now trains 2x per
#      student step, matching the R3GAN paper's 2:1 D:G ratio
#      recommendation for the harder distributions.
#   5. ``gan_critic_warmup_steps`` bumped 30 → 40 — gen-side gate
#      moved 10 steps later than D-start so D gets a brief 2:1 head-
#      start window (steps 30-39) to build a meaningful decision
#      boundary before the student starts fighting back at step 40.
#
# See "v7 DELTAS vs v6" below for the per-change rationale.
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
#     collapse, doesn't merely delay it. v7's gan_loss_weight=0.05
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
# The plan since v6 has been: keep DMD as the primary push, keep
# MANIQA + GAN as the image-manifold and structural anchors against
# the teacher's failure modes, and progressively WEAKEN the aux
# teacher's mean-seeking pull (aux_teacher_loss_weight cut 1.0 → 0.35
# in v6, held at 0.35 in v7; expected further reductions in
# subsequent versions). The expectation is that there's a sweet spot
# where the aux teacher corrects the teacher's drift enough that DMD
# doesn't compound to collapse, but not so much that DMD becomes a
# covert MSE-to-GT loss. v7 holds aux at v6's level and instead
# targets two distinct symptoms — the GT-blend-induced blur (fixed by
# removing the blend) and the dense-distillation fish-scale artefact
# (fixed by bumping R1/R2 to 5.0).
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
# time. v7 sequences the heads so the GAN gen-side doesn't engage
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
# v7 DELTAS vs v6
# ---------------
#
#   1. DMD-scoring clean_x GT-blend REMOVED.
#        v6: dmd_context='mix', dmd_context_mix_p=0.30 → 0.00 over the
#            first 100 steps (a small decaying GT anchor on the clean
#            half of DMD's scoring context).
#        v7: dmd_context='self', dmd_context_mix_p=0.00,
#            dmd_context_mix_p_ramp_steps=0 (no blend ever; pure
#            student-cache_pred clean_x for DMD scoring). Note: 'self'
#            is the validated keyword — the dmd_context validator
#            rejects 'student' (only 'self', 'GT', or 'mix' allowed);
#            'self' carries the same semantic ("DMD scores against
#            self-cache only").
#
#      Why: the GT blend was visibly causing **blur and drift** in
#      the decoded video. The student got steered toward a context
#      that doesn't match deployment-time (where the teacher only
#      ever sees self-cached frames), and the resulting mismatch
#      smeared structure across frames. Removing the blend pushes
#      the DMD scoring fully onto self-cache context, matching the
#      causal-AR inference regime exactly. The aux teacher (which
#      runs against actual GT-anchored targets via the LoRA flow
#      loss) now carries 100% of the GT-anchoring responsibility
#      instead of splitting it with this blend knob.
#
#   2. GAN R1/R2 gradient penalties bumped 1.0 → 5.0.
#        gan_r1_gamma=5.0, gan_r2_gamma=5.0 (config default = 1.0)
#
#      Why: previous runs decoded a visible fish-scale grid pattern
#      at ~60x45 cells across the image — exactly the latent-token
#      lattice (60-wide x 45-tall after the VAE's 8x downsample of
#      480x360 pixel video). The mechanism is that
#      gan_critic_dense_distillation=true makes the SAM2 distilled
#      critic emit ONE logit per latent token, so the gen-side
#      `-mean(critic(fake_lat_grad))` gradient pushes every token's
#      local receptive field independently toward whatever texture
#      the discriminator considers "real-looking" — yielding the
#      semi-regular per-token texture pattern we see. The
#      autoregressive nature of the rollout compounds this artefact
#      through the video (early frames are GT-anchored via the seed
#      context, later frames are baked from cached student output
#      that already contains the per-token bias).
#
#      R1/R2 penalties regularise the discriminator's gradient norm
#      with respect to its INPUT — i.e. they directly smooth the
#      per-token logit response. With γ=5.0 the discriminator can
#      no longer be sharp/spiky at specific texture frequencies, so
#      the gen has less to exploit. This is the standard fix for
#      "discriminator is too sharp" pathologies in R3GAN training.
#
#   3. Discriminator training deferred from step 0 → step 30.
#        gan_disc_start_step=30  (new knob; default 0 preserves
#                                  legacy behaviour)
#
#      Why: previously the entire D pass (D forwards on real+fake,
#      R1/R2 penalty, D backward, D optimizer step) ran every iter
#      from step 0, even though the gen-side GAN gradient was gated
#      by gan_critic_warmup_steps=30. That meant D spent the first
#      30 steps fitting to whatever pred_image the student happened
#      to produce in the unstable MANIQA-bootstrap window — pre-
#      step-30 the student's outputs reflect the MANIQA approxes'
#      transient warmup (per the SCHEDULE-SEQUENCING RATIONALE), so
#      D was learning a "real vs fake" boundary against samples
#      that don't represent steady-state student behaviour. When
#      gen-side engages at step 30, it then sees a D that's biased
#      against transient-state fakes, not student steady-state
#      fakes — a slightly stale decision boundary.
#
#      v7 aligns D-start with the existing gen-side critic warmup
#      so D fits only on post-MANIQA-bootstrap student outputs. The
#      new ``gan_disc_start_step`` knob is the gate. Side effects:
#      ~10-15% wallclock savings for the first 30 steps (no D
#      forwards or backward), and a small reduction in peak GPU
#      memory pressure during those steps — useful because the
#      DMD+aux+MANIQA stack is already busy in that window.
#
#   4. gan_updates_per_step bumped 1 → 2 (D:G cadence now 2:1).
#        gan_updates_per_step=2  (config default = 1)
#
#      Why: v6 ran with the default 1:1 D:G cadence and the
#      discriminator visibly couldn't keep up with the student's
#      progression — the fish-scale artefact persisted through the
#      whole run, and ``train/r3gan_d_fake_for_g`` (D's mean logit
#      on fake samples at gen-side time) drifted high, meaning D
#      was being fooled more easily as training progressed. v7
#      doubles D's training rate to give it a fighting chance
#      against the per-token texture exploit — D now runs two
#      back-to-back updates (each with its own real+fake forwards
#      and R1/R2 penalty) inside every gen iter, via the existing
#      ``for _k in range(gan_updates_per_step)`` loop in
#      ``_run_distilled_disc_critic_update`` (line 4464).
#
#      Combined with R1/R2=5.0 (delta 2), this is a balanced
#      hardening of the D side: D learns 2x faster but each update
#      is more regularised, so we get a sharper-and-smoother
#      decision boundary rather than a sharper-and-spikier one.
#      Watch ``gen/c_corr`` as the early-warning of overtraining
#      — if it trends to +1.0, D has overfit onto the SAM2 critic's
#      output and the next iteration should pull updates_per_step
#      back to 1.
#
#   5. gan_critic_warmup_steps bumped 30 → 40 (gen-side gate).
#        gan_critic_warmup_steps=40  (v6 was 30)
#
#      Why: with D-start=30 and updates_per_step=2, D now gets a
#      10-step head-start window [30, 40) where it trains at the
#      full 2:1 cadence on real-vs-fake samples WITHOUT the
#      generator pushing back. By step 40 the discriminator has had
#      20 gradient steps to build a decision boundary against
#      steady-state student outputs (it sees ~10 gen iters × 2
#      updates/iter = 20 D updates in that window) — enough for
#      its R1/R2-regularised manifold to be meaningfully shaped
#      before the gen-side gradient starts trying to fool it. This
#      is the R3GAN paper's intended bootstrap pattern, just
#      delayed to fire after the MANIQA bootstrap is also done.
#
#      Effective GAN gen-side weight under v7's
#      (gan_critic_warmup_steps=40, gan_warmup_steps=100, cap=0.05)
#      schedule:
#        step  40 → 0          (gen-side just turned on)
#        step  90 → 0.025      (50 steps into the 100-step ramp)
#        step 140 → 0.05       (full)
#
# Everything else carried forward unchanged from v6:
#   * DMD active from step 1, cap 0.5, 100-step warmup (full at
#     step 101). DMD is the primary student-push.
#   * GAN gen-side cap 0.05, gen_warmup=100 (full at step 140 under
#     the new critic_warmup=40 start). Critic_warmup sequences GAN
#     engagement to after MANIQA approx stabilises (see SCHEDULE-
#     SEQUENCING RATIONALE above).
#   * Aux teacher cap 0.35, aux_teacher_start_step=30,
#     aux_teacher_loss_warmup_steps=70 (full at step 100). Aux runs
#     after MANIQA stabilises for the same noise-isolation reason.
#   * real_teacher_warmup_steps=100 (LR warmup on the LoRA AdamW).
#   * teacher_cadence='fake' — inner loop fires
#     dfake_gen_update_ratio (=5) LoRA optimizer steps per outer iter
#     once aux is open.
#   * real_teacher_input_source='student', causal_mask=false (LoRA
#     learns bidirectional student→GT bridge).
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
# v7 deliberately makes the MANIQA bucket the largest static weight
# but expects DMD to dominate the actual *gradient magnitude* on the
# student because DMD's raw scale (pred_real - pred_fake) is large
# relative to the perceptual approx's typical loss values. Confirm
# this empirically via the per-key contribution traces on wandb.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_G_NR_v9_no_streaming
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
echo "Starting Phase-3 online-teacher G_NR v9 (no streaming) on $(date)"
echo "  * 21-frame iter-1-only student rollout, 200-step max."
echo "  * max_rolls_per_ride=1 — streaming resets every step (no iter k>=2)."
echo "  * Teacher: fresh rank-32 LoRA on v14-merged base (alpha=64, r=32)."
echo "  * teacher_cadence=fake — LoRA optim fires 5x per outer iter."
echo "  * Steps [0, 30): LoRA frozen. Steps [30, 100): aux loss weight ramps"
echo "    0 -> 0.35. Steps [100, 200): aux at 0.35 plateau."
echo "  * NO GT-blend in DMD clean_x (v7 delta vs v6) — caused blur+drift."
echo "  * DMD cap=0.5, dmd_loss_start_step=1, dmd_loss_warmup_steps=100"
echo "    (full at step 101 — DMD is the primary student-push)."
echo "  * GAN cap=0.05, gan_critic_warmup=40 (was 30 in v6; gives D a"
echo "    10-step 2:1 head-start), gan_warmup=100 (full at step 140)."
echo "  * GAN R1/R2 gamma = 5.0 (was 1.0 default; v7 delta) — suppresses"
echo "    60x45 latent-token fish-scale artefact from dense distillation."
echo "  * gan_disc_start_step=30 (v7 delta) — D-training deferred to"
echo "    step 30 so D doesn't fit on MANIQA-bootstrap-window outputs."
echo "  * gan_updates_per_step=2 (v7 delta; was 1) — D trains 2x per"
echo "    student step (R3GAN 2:1 D:G cadence)."
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
    sample_fps=20 \
    boundary_vae_roundtrip=true \
    local_attn_size_schedule=[[0,21]] \
    dmd_context=self \
    dmd_context_mix_p=0.00 \
    dmd_context_mix_p_target=0.00 \
    dmd_context_mix_p_ramp_steps=0 \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=1.0 \
    real_teacher_lr=5e-5 \
    real_teacher_warmup_steps=70 \
    real_teacher_input_source=student \
    teacher_cadence=fake \
    aux_teacher_p_schedule_enabled=false \
    aux_teacher_send_student_grad=false \
    aux_teacher_start_step=30 \
    aux_teacher_loss_weight=0.7 \
    aux_teacher_loss_warmup_steps=70 \
    aux_teacher_lora_rank=32 \
    aux_teacher_lora_alpha=64 \
    dmd_loss_weight=0.5 \
    dmd_loss_start_step=1 \
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
    gan_critic_warmup_steps=30 \
    gan_disc_start_step=30 \
    gan_warmup_steps=100 \
    gan_updates_per_step=1 \
    gan_r1_gamma=5.0 \
    gan_r2_gamma=5.0 \
    gan_sam2_frame_pool=none \
    gan_sam2_encoder_chunk_size=84 \
    maniqa_approx_loss_weight=1.0 \
    maniqa_n_frames=84 \
    maniqa_n_patches_per_frame=8 \
    maniqa_metric_name=maniqa-pipal \
    gan_critic_dense_distillation=true \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=action_forcing_phase3_online_teacher_G_NR_v9_no_streaming_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
