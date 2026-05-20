#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-NR-G-v27B
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
# Phase-3 online_teacher_NR_G_v27B: v27 + LEARNED FORWARD-NOISER for
# the CARN (causal-AR-noise) increment, replacing the v21 alt-head's
# failed application path.
#
# v27B vs v27 — knob deltas
# -------------------------
#   forward_noiser_enabled               : false -> true  (NEW)
#   forward_noiser_hidden_dim            : -     -> 512
#   forward_noiser_num_blocks            : -     -> 4
#   forward_noiser_max_carn_step         : -     -> 8
#   forward_noiser_loss_weight           : -     -> 1.0
#   forward_noiser_apply_in_aux          : -     -> true
#   forward_noiser_lr                    : -     -> 5e-5
# All other v27 knobs preserved verbatim.
#
# THE PROBLEM v27B SOLVES
# -----------------------
# Diagnostic data from v21..v28 wandb showed
# ``fake_alt_causal_AR_dir_rms = 0.0`` across the entire run history —
# meaning the alt-head's application output is ≈ GT every iter. The
# online real_score's LoRA has therefore been training on TRIVIAL GT
# denoising (input ≈ noise(GT), target = GT), not on the causal-AR-
# shaped inputs it sees at DMD inference. That's why pred_real_lora
# always looked sharp (it sees its training distribution) and pred_real
# was blurry (DMD inference distribution diverges from the LoRA's
# training).
#
# Root cause: the alt-head is a small output projection on the
# fake_score backbone, trained on noise(student_chunk) and applied to
# noise(gt_target). Backbone is trained as a student-distribution
# denoiser; applied to clean GT it produces ≈ denoised(GT) ≈ GT. The
# small alt-head projection has no capacity to inject AR-drift that
# isn't already in the backbone's representation.
#
# v27B FIX — LEARNED FORWARD-NOISER
# ---------------------------------
# Dedicated 3D ConvNet (NOT sharing fake_score's backbone) trained
# explicitly on the rollout1→rollout2 mapping. Rollout 2 has exactly
# +1 CARN step at each temporal position vs rollout 1, so learning
# this mapping = learning one CARN-noise step. Architecture:
#   * ~30M params @ hidden_dim=512, 4 ResBlocks
#   * Factorized spatial (1x3x3) + temporal (3x1x1) 3D convs
#   * FiLM-style CARN-step conditioning (sinusoidal embed + per-block
#     scale/shift) so the noiser learns step-dependent drift
#   * Zero-init final projection → starts as identity F(x, n) = x
#
# At apply time in the aux-teacher path, F is applied iteratively to
# GT chunks, each chunk stepped up to its target CARN level. Real_score
# is then trained on a noisy(causal_AR_GT) input distribution that
# matches what it sees at DMD inference. Replaces the v21 alt-head
# causal_AR_x0 path (which is auto-disabled when forward_noiser is
# applied each iter).
#
# AUDIT FIXES BAKED IN (from auditor review of initial v27B impl)
# --------------------------------------------------------------
#   1. ForwardNoiser DDP uses find_unused_parameters=True (defensive
#      against rank-divergent early-return paths in the noiser loss).
#   2. Explicit CARN_step=0 training pair: rollout 1 seed-last-chunk
#      (clean GT) → rollout 2 anchor. The streaming critic window
#      never covers this transition on its own; we inject it per iter
#      so the noiser's first iteration (every chunk starts at CARN=0
#      at application time) is well-trained.
#   3. Visualization: clean_x_aux + aux_noise_base stashed for the
#      eval video logger so we can visually verify what real_score's
#      training input looks like each iter (should evolve from
#      ≈GT toward AR-noisy as the noiser learns).
#
# LEGACY v23 CONTEXT (anti-collapse knobs preserved from v27)
# -----------------------------------------------------------
#
# v23 vs v22 — knob deltas
# ------------------------
#   anti_collapse_loss_weight   : 1.0 -> 0.2   (softer std floor)
#   anti_collapse_mean_weight   : 1.0 -> 0.0   (mean anchor OFF)
# All other v22 (= v21) knobs preserved verbatim.
#
# Why drop the mean anchor and lower the std-floor weight:
# Running v22 with both terms at 1.0 (matching the 14B_teacher_G_v2
# precedent) put a relatively strong latent-space prior on top of
# the alt-head + DMD + GAN + MANIQA chain. v23 backs off to test
# whether the std floor at 0.2 (a quarter of v22's strength) is
# enough on its own to prevent gray-collapse without the mean
# anchor's pull toward 0 (which assumed VAE prior is zero-centered
# — only true in aggregate, not per-frame).
#
# v22 vs v21 (for historical context) — both terms had been added:
#   anti_collapse_loss_weight   : 0.0 -> 1.0   (std floor)
#   anti_collapse_mean_weight   : 0.0 -> 1.0   (mean anchor)
# v23 keeps the std floor at a lower weight and removes the mean
# anchor entirely.
#
# WHAT THE TWO ANTI-COLLAPSE TERMS DO
# -----------------------------------
# Both terms fire inside ``_compute_kl_grad`` whenever ``gt_target``
# is available at the DMD call site (i.e. every gen-step loss path
# that supplies gt_latents). They are added DIRECTLY to ``dmd_loss``
# after the main MSE term, so they take effect from step 0 unless
# the wider DMD weight gate is closed; both work in latent space, no
# VAE roundtrip.
#
#   anti_collapse_loss_weight (STD FLOOR — gray-collapse guard)
#     loss = ReLU(std(GT) − std(pred_x0))^2  per-(batch, frame)
#     Penalises ONLY the direction where the student's pred_x0 std
#     contracts below the GT frame's natural spread. If the student
#     produces a flatter-than-GT latent — the textbook gray-collapse
#     signature observed at step ~30–46 of the 14B-teacher runs —
#     this term pushes the std back up. No penalty when std is
#     already at-or-above GT (one-sided ReLU).
#
#   anti_collapse_mean_weight (MEAN ANCHOR — white-collapse guard)
#     loss = mean(pred_x0)^2  per-(batch, frame)
#     The std floor alone has a degenerate solution: inflate latent
#     magnitude (translate the distribution far from zero) and the
#     variance increases without changing shape. The VAE decodes
#     translated-large-magnitude latents as saturated/white pixels —
#     this is the overshoot of gray-collapse seen in 14B_teacher_G
#     j4625936. The mean anchor pulls per-frame channel-pooled mean
#     toward 0 (the VAE's zero-centered prior), so std cannot be
#     satisfied by translation. Soft, scalar-only — does NOT
#     supervise pixel-wise. Critically, it does NOT overlap with
#     gt_latent_mse/mae (those remain OFF in v22) which would do
#     full pixel-wise GT supervision and overpower DMD.
#
# WHY BOTH AT WEIGHT 1.0
# ----------------------
# The 14B_teacher_G_v2 precedent uses 1.0 for both. The std floor at
# 1.0 was empirically enough to prevent gray-collapse without
# distorting the DMD-favoured shape; the mean anchor at 1.0 was
# chosen to match (and the override block in that precedent likewise
# sets both to 1.0 — the doc body's earlier ``0.3`` note is stale).
# Both losses are squared and quadratic in the deviation, so they
# saturate gracefully — no risk of overpowering DMD when the
# distribution is healthy (deviation ≈ 0 → loss ≈ 0).
#
# INTERACTION WITH v21'S ALT HEAD
# -------------------------------
# These guards operate on the student's pred_x0 in the main DMD
# loss; they are independent of the alt head, the EMA-real path,
# the aux teacher LoRA, and the AR-shaped-noise aux pass. They
# constrain WHERE in latent space the student is allowed to live;
# the alt head controls WHAT direction the LoRA learns to denoise.
# Orthogonal mechanisms, no expected interference.
#
# ---- LEGACY v21 DOC BELOW ----
#
# Phase-3 online_teacher_NR_G_v21: fake_score gains a second head trained
# to predict the EMA-real teacher's x0. The alt head shares fake_score's
# WAN backbone (single forward → both outputs) but has its own output
# projection. Its input is detached inside the model so its loss
# never propagates into the backbone — the alt head's params train
# exclusively on the alt loss against pred_real_EMA.
#
# v21 PRINCIPLE (per user thesis)
# -------------------------------
# v20's NoiseAuxLite + (chunk - pred_real) target turned out to be
# DMD-direction in disguise — it just amplified what DMD was already
# doing. The genuine *causal autoregressive* denoising signal is the
# small AR-noise reduction direction that fake_score has already
# accumulated relative to the AR-blind GT-supervised real_score. We
# isolate it by training an alt head on fake_score's backbone to
# predict pred_real_EMA (the EMA-real's x0 estimate). At aux-teacher
# time, applying the alt head to a noised GT produces a "causal AR GT"
# estimate (= GT shaped through the older-teacher's predictions). The
# real_score's aux teacher pass then trains on:
#     noisy_input = add_noise(causal_AR_GT, eps, t)
#     target_flow = eps - GT_TRUE
# so the LoRA learns to denoise AR-shaped noise back to TRUE GT — a
# learning signal that DMD CANNOT provide (DMD only knows the
# fake-vs-real direction, not the causal-vs-AR-blind direction).
#
# v21 STAGED SCHEDULE
# -------------------
# v26 SCHEDULE — fronted (no EMA-real warmup needed)
# --------------------------------------------------
# v26's alt-head targets rollout2_student (v24 mode), not ema_real_x0,
# so there's no EMA-real divergence window to wait for. The rollout2
# target is meaningful from iter 1, so alt training AND aux-teacher
# consumption start at step 0.
#  step 0     : ``fake_alt_head_start_step`` AND ``fake_alt_apply_
#               start_step`` — alt loss trains from iter 1, and the
#               aux teacher pass starts consuming alt's output as
#               noise_base immediately. Real_score sees AR-shaped
#               noised input from step 0; target stays true GT.
#  step 20    : ``dmd_loss_start_step`` — DMD gen-side gradient turns
#               on, ramps over 150 steps.
#  step 40    : ``gan_disc_start_step`` = ``gan_critic_warmup_steps``
#               — GAN engages, gen-side ramps quadratically over 25
#               steps (= plateau at step 65).
#
# v21 vs v20 — knob deltas
# ------------------------
#   noise_aux*                        : RIPPED (gone from codebase)
#   aux_teacher_ar_noise_*            : RIPPED (gone from codebase)
#   real_score_ema_weight: 0.99       -> 0.95   (faster EMA divergence)
#   gan_disc_start_step: 60           -> 80     (DMD warms first)
#   gan_critic_warmup_steps: 60       -> 80
#   fake_alt_head_enabled             -> true   (NEW)
#   fake_alt_head_start_step          -> 20     (NEW)
#   fake_alt_apply_start_step         -> 40     (NEW)
# All other v18/v20 knobs preserved.
#
# LEGACY v20 DOC BELOW
# --------------------
# (Phase-3 online_teacher_NR_G_v20: v18 + learned noise_aux model.
#
# v17/v18/v19 POST-MORTEM
# -----------------------
# v17 (ref=gt), v18 (ref=pred_real), v19 (ref=pred_real_lora) all
# collapsed visually after step ~50-100. Root cause: the pairwise
# residual (chunk - reference_x0) at rollout positions where chunk
# and reference share a temporal shift relative to GT — the residual
# captures BOTH genuine AR drift AND the time-warping artefact,
# blurring the teacher's learned direction.
#
# v20 STRATEGY — LEARNED FORWARD-NOISING MODEL
# --------------------------------------------
# Replace the pairwise residual with a small learned ConvNet
# ``NoiseAuxLite`` (~8M params @ hidden=192, num_blocks=4) trained
# online on the forward direction:
#   input  = pred_image (= chunk, detached)
#   target = 2 * pred_image - pred_real  (= pred_image + AR_noise)
# At aux-teacher time (no_grad), we feed GT through it:
#   ar_residual = noise_aux(gt_target) - gt_target
# This is well-defined per-sample on the forward (clean -> noised)
# direction and avoids the time-warping problem of inverse pairs.
# Warmup: aux_teacher_ar_noise_reference falls back to no-AR for the
# first ``noise_aux_warmup_steps`` (=50) steps while noise_aux trains.
#
# v20 vs v18 — only the AR reference + noise_aux toggle differ
# ------------------------------------------------------------
#  aux_teacher_ar_noise_reference:  pred_real -> noise_aux
#  noise_aux_enabled:                false -> true
#  noise_aux_warmup_steps:           -      -> 50
#  noise_aux_hidden_channels:        -      -> 192
#  noise_aux_num_blocks:             -      -> 4
#  noise_aux_lr:                     -      -> 1e-4
# All other v18 knobs unchanged.
#
# LEGACY v16 DOCS BELOW (kept for historical context)
# ---------------------------------------------------
# Phase-3 online_teacher_NR_G_v16: v15 CONFIG + GT-LATENT RECON OFF.
# Isolates whether GT-recon (MSE + MAE pull toward GT) was the cause
# of fast-forward / mean-collapse in v13/v14/v15.
#
# v15 POST-MORTEM — STILL FAST-FORWARD AT STEP 16
# -----------------------------------------------
# v15 reverted four v13/v14 changes (max_gradient_chunks 2->0,
# dfake_ratio 2->5, AR-noise combined->off, gan_loss_weight 0.3->0.5)
# while keeping CFG-off + v14 LRs + dmd_start=40 + GT-recon. The
# user cancelled v15 at step 16 — visually fast-forwarded and
# mean-collapsed. At step 16 the ONLY student-training losses are:
#   - MANIQA approx perceptual (full strength by step 15)
#   - GT-latent MSE + MAE recon (constant from step 0)
# DMD weight is 0 at step 16 (start=40), GAN gen-side is 0 (critic
# warmup=60), aux teacher has send_grad=false (no gen gradient).
# This DEFINITIVELY rules out the teacher as the cause of the early
# collapse.
#
# That leaves three candidates: (a) MANIQA approx, (b) GT-recon,
# (c) the doubled gen LR (2e-5) hitting whichever loss harder.
# v16 zeros out (b) to test: if v16 doesn't fast-forward at step 16,
# GT-recon was the culprit. If it still does, it's MANIQA or LR
# (v17 will isolate those).
#
# v16 vs v15 — one knob change
# ----------------------------
#  gt_latent_mse_loss_weight: 0.1 -> 0.0
#  gt_latent_mae_loss_weight: 0.05 -> 0.0
# All other v15 knobs unchanged: max_gradient_chunks=0,
# dfake_gen_update_ratio=5, AR-noise mode=off, gan_loss_weight=0.5,
# CFG off, lr=2e-5, real_teacher_lr=7.5e-5, dmd_loss_start_step=40.
#
# v16 PREDICTION
# --------------
# Most likely: v16 looks like v9 visually (starts well, slow blur
# over many steps), because MANIQA + the 2x LR are still pulling
# toward "high-MANIQA-score generic outputs". If v16 has v9-style
# slow blur instead of v15's step-16 fast-forward, GT-recon was
# the dominant cause and we restore MANIQA-only training as a
# baseline in v17.
#
# v15 LEGACY DOCS BELOW
# ---------------------
#
# Phase-3 online_teacher_NR_G_v15: REVERT TO v12 STRUCTURAL BASE +
# KEEP v14 LR/CFG IMPROVEMENTS. Targets the "fast forward / mean
# collapse" failure mode unique to v13/v14.
#
# VISUAL OBSERVATIONS (sample videos, post-run inspection)
# --------------------------------------------------------
#   v9  — starts well, eventually goes blurry. Slow mean collapse.
#   v10 — like v11 but more coloured. Less collapsed than v11.
#   v11 — starts well, blurs quickly, then devolves into BLUE.
#         Channel-skewed collapse (peculiar blue tint).
#   v12 — coloured + a bit blurry + SMOOTH MOTION TRAILS. Best run
#         visually; the motion-trail signature means the student
#         is correctly tracking temporal dynamics chunk-to-chunk.
#   v13 — "on fast forward" — temporal dynamics ACCELERATED relative
#         to GT. New failure mode not seen in v9-v12.
#   v14 — blurry and gray at step 46. Faster mean collapse than v13.
#
# Loss-curve picture was misleading: v13 had the cleanest sustained
# c_corr (>= +0.88 for 6 windows), but the actual rollout videos show
# a regression vs v12's "coloured + motion trails". The diagnostics
# we log (gen_loss, dmd_grad, c_corr) don't measure temporal
# coherence or color distribution — they can't catch this.
#
# WHAT'S COMMON BETWEEN v13/v14 (BAD) AND DIFFERENT FROM v12 (GOOD)
# ----------------------------------------------------------------
# Four config changes simultaneously introduced at v13:
#   (a) max_gradient_chunks=2 — gradient on 6 of 21 frames; chunks
#       3-6 receive NO training signal at all
#   (b) dfake_gen_update_ratio=2 (was 5) — fake_score chases more
#       slowly
#   (c) AR-noise mode=combined + chunk-GT residual (was two_pass +
#       chunk-pred_real in v11/v12)
#   (d) real_guidance_scale=0 (CFG off; was 1.0 in v9-v12)
#
# Most likely culprit for "fast forward / mean collapse" is (a):
# with no training signal on chunks 3-6, those frames are
# unconstrained at training time and the student extrapolates
# whatever motion vector the early chunks suggest — appearing as
# accelerated time. The teacher's pred_real on those frames is also
# stale (LoRA never learned to denoise them), reinforcing the lack
# of constraint. (b) and (c) compound: slower fake_score + AR-noise
# that explicitly captures temporal drift, with no late-chunk
# correction. (d) is plausibly benign — CFG off doesn't change the
# qualitative training contract, just removes a forward pass.
#
# v15 STRATEGY — revert (a) and (b), keep (c) and (d), keep v14 LRs
# -----------------------------------------------------------------
#  (R1) max_gradient_chunks=2 -> 0 (unlimited). All 21 frames get
#       training signal again, matching v12. Late chunks regain
#       gradient supervision so the teacher learns to denoise them
#       and the student doesn't extrapolate freely.
#  (R2) dfake_gen_update_ratio=2 -> 5. Pairs with R1 — restoring
#       the v12 fake_score cadence so it can track the now-larger
#       gradient window properly.
#  (R3) gan_loss_weight=0.3 -> 0.5. Match v12. With full gradient
#       window restored, more GAN pressure lands effectively.
#  (R4) aux_teacher_ar_noise_mode=combined -> off. With max_gradient_
#       chunks reverted, AR-noise needs to be re-validated. Turning
#       it off makes v15 the cleanest comparison to v12 — only
#       CFG-off and LR bumps differ. If v15 recovers v12's quality,
#       we re-enable AR in v16.
#  KEEP from v13/v14:
#       - real_guidance_scale=0 (CFG off): no observed harm
#       - lr=2e-5 (gen 2x), real_teacher_lr=7.5e-5 (1.5x)
#       - dmd_loss_start_step=40 (earlier DMD onset)
#       - gt_latent_mse_loss_weight=0.1, gt_latent_mae=0.05
#
# v15 PREDICTION (write here so we can score the hypothesis)
# ----------------------------------------------------------
# If max_gradient_chunks=2 was the dominant cause, v15 should look
# like v12 (coloured + motion trails) but reach higher quality
# sooner thanks to v14's LR/start tuning. If v15 still has fast-
# forward, the culprit is (c) — chunk-GT AR noise — and v16 reverts
# that too.
#
# ---- v14 LEGACY CONTEXT BELOW ----
#
# v14 vs v13 — four knob changes (LR + start_step retuning)
# ---------------------------------------------------------
# v13's first 60 steps showed a markedly gentler regime than v9-v12:
#   step 51 gen_loss:  v6 +0.35,  v7 +0.37,  v9 +0.41,
#                      v10 +0.41, v12 +0.58, v13 +0.099 (4-6x lower)
#   step 51 dmd_grad:  v9-v12 ~0.95, v13 0.40 (half)
# This is because max_gradient_chunks=2 (gradient on 6/21 frames) +
# dfake_gen_update_ratio=2 (fake_score chases 2.5x slower) + new
# AR-noise anchor combine to dramatically reduce the per-step
# training pressure. v13 is structurally anti-collapse but at the
# cost of being UNDER-TRAINED within the 2h walltime — projected
# end of run is step ~170/250, DMD at 73% of plateau.
#
# v14 compensates with more aggressive LRs and an earlier DMD start:
#  (1) lr (gen): 1e-5 (YAML default) -> 2e-5
#      Doubled gen LR so the student converges faster despite the
#      smaller effective gradient window.
#  (2) real_teacher_lr: 5e-5 -> 7.5e-5
#      1.5x faster LoRA real_score so it adapts to the AR-augmented
#      input distribution sooner. fake_lr stays at default (1e-5)
#      so dfake_gen_update_ratio=2 remains effective at slowing the
#      fake_score chase relative to the gen.
#  (3) dmd_loss_start_step: 60 -> 40
#      Earlier DMD onset. With v13's anti-collapse mechanisms
#      (AR-noise teacher anchor, GT-recon constants, max_gradient_
#      chunks=2 reducing exposure to noise-dominated late chunks),
#      the system can tolerate DMD signal sooner. Reaches DMD
#      plateau ~30 steps earlier, useful within 2h walltime.
#  (4) gan_loss_weight: 0.2 -> 0.3
#      Modest 1.5x bump on GAN. v13's reduced fake_score chase
#      slightly weakens the DMD signal; a slightly stronger GAN
#      compensates without re-introducing v10's over-GAN regime.
#
# Everything else carries over from v13 unchanged: max_gradient_
# chunks=2, dfake_gen_update_ratio=2, AR-noise combined mode at
# scale=1.0 with α(t)=clamp((1-σ)/σ,10), seed_mask_chunks=1,
# burn_in_steps=20, variance_renorm=true (decay 0.99), CFG off,
# gt_latent_mse=0.1, gt_latent_mae=0.05.
#
# v13 LEGACY DOCS BELOW (kept for context)
# ----------------------------------------
#
# v13 vs v10 — three structural changes
# -------------------------------------
#  (A) CFG off: real_guidance_scale 1.0 -> 0.0. The teacher's
#      conditional forward is now used directly with no CFG
#      extrapolation. Cheaper, simpler, removes the (cond - uncond)
#      contribution from every teacher output.
#
#  (B) gan_loss_weight 0.5 -> 0.2 (lighter GAN). With AR-noise
#      anchoring the teacher into the student's actual distribution,
#      we expect to need less raw GAN pressure to stay on-manifold.
#
#  (C) AR-noise-aware aux teacher training. The teacher is no longer
#      trained only on add_noise(GT, ε, t); per iter the aux pass
#      now also sees a clean-side input perturbed by the empirical
#      AR-drift of the student rollout:
#         ar_residual = (chunk - gt_target).detach()        [x0-space]
#         α(t)        = clamp((1 - σ_t) / σ_t, max=α_max)    [flow conv]
#         ar_norm     = ar_residual / EMA(rms(ar_residual))  [renorm]
#         seed_mask   = linear ramp 0->1 over first K chunks  [drift-aware]
#         burn_in     = min(1, step / N)                      [warmup]
#         noise_base_ar = gt + scale·burn_in·α(t)·seed_mask·ar_norm
#         noisy_input   = add_noise(noise_base_ar, ε, t)
#      Target stays ``ε - gt_target`` (pure GT), so the teacher
#      learns to denoise drift-perturbed inputs back toward
#      unperturbed GT. Mode=combined (single forward), scale=1.0,
#      α_max=10, seed_mask_chunks=1, burn_in_steps=20,
#      variance_renorm=true (EMA decay 0.99).
#
#      Rationale for using (chunk - GT) instead of (chunk - pred_real)
#      from v11/v12: GT is the actual reference at the seed-adjacent
#      chunks; the residual is guaranteed to be AR drift there. As
#      the rollout extends, drift compounds and GT stops being a
#      faithful predictor of the student — but (D) below restricts
#      the gradient window to the seed-adjacent rolls anyway, so the
#      late-rollout-drift concern doesn't bite.
#
#  (D) max_gradient_chunks=2. Only the first two rolled chunks
#      (frames 0..5) get gradient for DMD score, aux teacher, and
#      critic losses. Later chunks are noise-dominated and discarded.
#      Composes with the existing v14-LoRA last-chunk boundary mask
#      (the intersection of "first 2 chunks" and "not last chunk").
#      Wired via the new ``_dmd_score_grad_mask`` cap.
#
# v9 POST-MORTEM — DOUBLED GAN PREVENTED COLLAPSE, RAN 2H FULL TERM
# ----------------------------------------------------------------
# v9 (gan_loss_weight=0.2) was a clear improvement over v7. It hit
# the 2h SLURM walltime at step ~210/250 without ever showing the
# hard mode-seeking collapse signature. dmd_grad held in 0.85-0.99
# the whole run (vs v7's drop to 0.66 at step 151, v6's 0.55 at
# step 81). gen_loss oscillated between -0.21 and +0.41 with
# consistent recovery (step 181 surged back to +0.343 like step 51).
#   step 51:  gen_loss +0.412, dmd_grad 0.91   (v6/v7 surge mirror)
#   step 81:  gen_loss -0.207, dmd_grad 0.90   (v6 collapse moment;
#                                               v9 dmd_grad stayed
#                                               healthy)
#   step 121: gen_loss -0.010, dmd_grad 0.96   (v7 gray-collapse
#                                               moment; v9 stable)
#   step 151: gen_loss +0.029, dmd_grad 0.98   (v7 hard collapse;
#                                               v9 clean recovery)
#   step 181: gen_loss +0.343, dmd_grad 0.92   (strong bounce-back)
#   step 201: gen_loss +0.098, dmd_grad 0.72   (briefly elevated
#                                               DMD pull but no
#                                               collapse signature)
# c_corr still oscillated between +0.97 and -0.99 (disc within-batch
# anti-alignment persisted) but the doubled GAN provided enough
# amplitude during aligned phases to keep the student stable through
# the anti-aligned phases. d_real-d_fake separation kept growing
# from 0.000 (step 61) to 0.16 (step 201) — disc IS learning on
# average even if per-sample ranking flips intermittently.
#
# v10 STRATEGY — PUSH GAN HARDER + ADD GT-LATENT ANCHOR
# -----------------------------------------------------
# Two changes vs v9:
#   (a) Bump gan_loss_weight 0.2 -> 1.0. This flips the head balance:
#       DMD:GAN plateau ratio was 2.5:1 in v9, becomes 1:2 in v10 (GAN
#       now dominates DMD by 2x at plateau). GAN is now the primary
#       student trainer; DMD becomes a secondary regulariser.
#   (b) Add constant-weight latent-space recon losses against GT:
#         gt_latent_mse_loss_weight = 0.1
#         gt_latent_mae_loss_weight = 0.05
#       Both fire every step from step 0 with no ramp or warmup —
#       a small direct anti-drift anchor pulling pred_image toward
#       the GT latent window. Independent of DMD/GAN/MANIQA chains
#       (no critic, no approx, no VAE — pure latent-space L1+L2).
#       Implemented in trainer ``_compute_gt_latent_recon_loss``.
# Accepted risk: during disc-anti-aligned phases the 5x stronger GAN
# pushes 5x harder in the wrong direction. v9 absorbed that shock
# because GAN was still below DMD; with GAN now > DMD, an extended
# anti-aligned phase could pull the student off-manifold faster than
# DMD can correct. If v10 collapses, the disc itself is confirmed
# as the bottleneck and the next lever is disc-side tuning
# (R1/R2 gamma up, longer disc head-start, lower disc LR) — NOT
# pushing GAN weight higher.
#
# Alternatively, if v10 holds without collapse, that's evidence the
# disc anti-alignment is a tolerable transient state and the system
# is GAN-bound rather than disc-bound at higher GAN weights.
#
# STAGE-BY-STAGE DESIGN (inherited from v7)
# -----------------------------------------
#
#   Stage 1 (steps 0-15):
#     * MANIQA-approx ramps in linearly (perceptual_approx_warmup=0,
#       perceptual_approx_ramp_steps=15). The approx itself is also
#       distilling against real MANIQA — its dense_loss drops fastest
#       through this window. Student-side signal is MANIQA only;
#       DMD off, GAN off.
#     * Online real_score (LoRA aux) trains from step 0
#       (aux_teacher_start_step=0).
#
#   Stage 2 (steps 20-60):
#     * DMD engages at step 20 (dmd_loss_start_step=20) and ramps to
#       its plateau 0.5 over 150 steps.
#     * MANIQA at plateau; LoRA still ramping (full at step 80).
#     * GAN still off.
#
#   Stage 3 (steps 60-end):
#     * GAN engages — D and gen-side both gate at step 60 simultane-
#       ously (no D head-start).
#     * Gen-side weight ramps via a QUADRATIC shape over 25 steps
#       (gan_warmup_steps=25, gan_warmup_shape=quadratic). Plateau
#       (1.0) at step 85.
#
# WHAT INFORMS THE STUDENT vs WHAT INFORMS THE LoRA
# -------------------------------------------------
# Two independent gradient flows. The student is updated by the gen
# optim step driven by MANIQA + DMD + GAN (each with its own ramp).
# The LoRA is updated by its own aux optim step driven by flow-loss
# against GT (aux_teacher_send_student_grad=false severs the LoRA
# aux pass from the student). v8 confirmed that flipping this to
# true causes deeper collapse — keep false. EMA on the LoRA params
# (weight 0.99) means DMD scoring reads a slow-moving target while
# the live LoRA tracks the latest student outputs.
#
# Key knobs and their effective values:
#
#   knob                                 value     comment
#   ---------------------------------    -------   ------------------
#   max_steps                            250
#   perceptual_approx_warmup_steps       0
#   perceptual_approx_ramp_steps         15        MANIQA full at 15
#   aux_teacher_start_step               0         LoRA trains from 0
#   aux_teacher_loss_warmup_steps        80        LoRA full at 80
#   aux_teacher_send_student_grad        false     (v8 collapsed when true)
#   real_score_ema_weight                0.99      slow EMA target
#   real_teacher_input_source            gt        proper score match
#   dmd_context                          self      no GT mix in DMD
#   dmd_loss_weight                      0.5
#   dmd_loss_start_step                  20
#   dmd_loss_warmup_steps                150       DMD full at 170
#   gan_disc_start_step                  60        D starts at 60
#   gan_critic_warmup_steps              60        gen-side gate at 60
#   gan_warmup_steps                     25        gen-side full at 85
#   gan_warmup_shape                     quadratic gentle but quick
#   gan_loss_weight                      1.0       <-- v10 change (5x v9, 10x v7)
#   gan_critic_dense_distillation        true
#   gan_updates_per_step                 1
#   gan_r1_gamma / gan_r2_gamma          2.0 / 2.0
#
# Effective head-weight timeline (GAN plateau is 1.0 vs v9's 0.2):
#
#   step   MANIQA   DMD       GAN gen   aux LoRA
#   ------ -------- --------- --------- ---------
#     0    0.000    0.000     0         0.000
#    15    1.000    0.000     0         0.188
#    20    1.000    0.000     0         0.250
#    60    1.000    0.133     0.000     0.750
#    61    1.000    0.137     1.6e-3    0.763
#    65    1.000    0.150     0.0400    0.813
#    70    1.000    0.167     0.1600    0.875
#    75    1.000    0.183     0.3600    0.938
#    80    1.000    0.200     0.6400    1.000
#    85    1.000    0.217     1.0000    1.000
#   135    1.000    0.383     1.0000    1.000
#   170    1.000    0.500     1.0000    1.000
#   250    1.000    0.500     1.0000    1.000
#
# DMD:GAN ratio at plateau is now 0.5:1.0 = 0.5 — GAN dominates DMD
# by 2x (v9 was 2.5, v7 was 5.0). This is a major rebalancing — the
# student is now primarily GAN-trained.
#
# NO-STREAMING MECHANISM (inherited)
# ----------------------------------
# max_rolls_per_ride = 1.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_NR_G_v27B
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "GAN_SAM2_CKPT=$GAN_SAM2_CKPT"
echo "GAN_SAM2_CFG=$GAN_SAM2_CFG"
echo "Starting Phase-3 online-teacher NR_G v23 (v22 minus mean anchor, softer std) on $(date)"
echo "  * v23 = v22 with:"
echo "      anti_collapse_loss_weight 1.0 -> 0.2 (softer std floor)"
echo "      anti_collapse_mean_weight 1.0 -> 0.0 (mean anchor OFF)"
echo "  * Std-floor term still fires from step 0 inside _compute_kl_grad"
echo "    when gt_target is available: ReLU(std(GT) - std(pred_x0))^2 (one-sided)."
echo "  * Independent of the alt head — operates on pred_x0 in latent space."
echo "  * v17/v18/v19 collapsed: pairwise (chunk - reference) residual"
echo "    suffered from time-warp blur at AR rollout positions."
echo "  * v20 replaces it with a learned forward-noising ConvNet:"
echo "      f(pred_image) -> 2*pred_image - pred_real"
echo "    At aux-teacher time, ar_residual = noise_aux(gt) - gt."
echo "  * noise_aux warmup_steps=50; before that, AR falls back to no-AR."
echo "  * ~8M params, fp32 ConvNet, AdamW lr=1e-4. Trained every gen step."
echo "  * v15 fast-forwarded by step 16 — DMD weight=0 there, so the"
echo "    teacher is not the cause. Suspects: MANIQA approx, GT-recon, 2x LR."
echo "  * v16 isolates GT-recon: gt_latent_mse_loss_weight 0.1->0,"
echo "    gt_latent_mae_loss_weight 0.05->0. All other v15 knobs unchanged."
echo "  * Prediction: if v16 doesn't fast-forward at step 16, GT-recon was"
echo "    the culprit. If it still does, v17 will test MANIQA / LR."
echo "  * Visual obs from v9-v14:"
echo "      v12 = best (coloured + motion trails); v13 = fast-forward;"
echo "      v14 = blurry/gray; v11 = blue collapse; v9/v10 = slow blur."
echo "  * v15 reverts (a) max_gradient_chunks 2->0, (b) dfake_ratio 2->5,"
echo "    (c) gan_loss_weight 0.3->0.5, (d) aux AR-noise mode combined->off."
echo "  * v15 keeps: CFG off (real_guidance_scale=0), lr=2e-5,"
echo "    real_teacher_lr=7.5e-5, dmd_loss_start_step=40, gt_latent recon."
echo "  * Prediction: if max_gradient_chunks was the dominant cause,"
echo "    v15 looks like v12 with v14's faster convergence."
echo "  * 21-frame iter-1-only student rollout, 250-step max."
echo "  * max_rolls_per_ride=1 — streaming resets every step (no iter k>=2)."
echo "  * Teacher: fresh rank-32 LoRA on v14-merged base, EMA(0.99) target."
echo "  * v13 NEW (vs v10):"
echo "      - CFG off (real_guidance_scale=0)."
echo "      - gan_loss_weight 0.5 -> 0.2 (lighter GAN; AR-noise carries weight)."
echo "      - aux_teacher_ar_noise_mode=combined; AR noise = (chunk-GT) fresh"
echo "        each iter, added to GT before noising (CLEAN-side perturbation)."
echo "        Flow-space scaling α(t)=clamp((1-σ)/σ, max=10), seed-mask ramp"
echo "        over first 1 chunk, 20-step burn-in, EMA-renormalised RMS."
echo "      - max_gradient_chunks=2: only first 2 rolled chunks (frames 0..5)"
echo "        carry gradient for DMD score AND aux teacher. Later chunks are"
echo "        noise-dominated and discarded."
echo "  * gt_latent_mse=0.1, gt_latent_mae=0.05 — direct anti-drift latent"
echo "    L1+L2 against GT, every step, no ramp (carried from v10)."
echo "  * v8 POST-MORTEM: aux_teacher_send_student_grad=true caused deeper"
echo "    collapse (gen_loss -0.36 at step 101 vs v7's +0.15) — kept false here."
echo "  * Stage 1 [0, 20): MANIQA ramps in (full at step 15). LoRA"
echo "    aux trains from step 0 (warmup_steps=80, full at step 80)."
echo "    DMD off, GAN off."
echo "  * Stage 2 [20, 60): DMD engages (start=20, warmup=150, cap=0.5)."
echo "    MANIQA at plateau; LoRA still ramping. GAN still off."
echo "  * Stage 3 [60, end): GAN engages with QUADRATIC ramp"
echo "    (gan_warmup_steps=25, shape=quadratic). D-side starts simul-"
echo "    taneously (no head-start). GAN full at step 85 (plateau 1.0)."
echo "  * Aux LoRA: aux_teacher_send_student_grad=false (LoRA aux is"
echo "    severed from the student; DMD is the sole student-grad path"
echo "    through the teacher). real_teacher_input_source=gt."
echo "    real_score_ema_weight=0.99."
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
    gan_loss_weight=0.2 \
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
    gan_warmup_shape=quadratic \
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
    anti_collapse_loss_weight=0.8 \
    anti_collapse_mean_weight=0.0 \
    max_gradient_chunks=3 \
    run_name=action_forcing_phase3_online_teacher_NR_G_v27B_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
