#!/bin/bash
# =====================================================================
# PIXDINO -- the gt_vs_fake discriminator's features come from DINOv2 on
# DECODED PIXELS, and its gradient reaches the generator through the
# distilled LATENT critic (WP-SURROGATE), never through a decode.
#
# Design + memory arithmetic + connection proofs:
#     analysis/gan_tuning/PIXEL_FEATURE_SOURCE.md
# Code: model/ladd_pixel_features.py, model/ladd_disc.py (pixel branch),
#     trainer ``_ladd_install_pixel_decode`` / ``_ladd_pixel_logs`` /
#     ``_pix_surrogate_grad_telemetry``.
#
# FORKED VERBATIM from sbatch/gansig_gtvf_dmd_carnpure.sbatch
# (dmd10k_gansig_gtvf_dmd_carnpure_j6130135). Every knob below the
# ``PIXDINO OVERRIDES`` banner is new; everything above it is that arm.
#
# ---------------------------------------------------------------------
# WHY. analysis/sharpness/TEXTURE_REVIEW.md 0.2 measured the "texture of
# dots" the researcher sees and identified it exactly: a 16 px 2-D
# lattice. fold2d P=16 residual peak-to-peak 11.51 luma units for
# ``ofclean`` against 3.30 for the GT half THROUGH THE SAME DECODER, with
# 39 % of the folded variance in the non-separable dot term against 13 %
# for GT and 4-11 % for every other arm. 16 px = 2 latent cells = ONE WAN
# TRANSFORMER PATCH.
#
# The LADD disc taps WAN blocks [0,2,4,8,29], each emitting one token per
# (1,2,2) patch, and ``LADDDiscHead`` emits one logit per token. The
# generator's adversarial gradient is therefore a field defined ON THE
# PATCH GRID and the VAE decoder prints it. That is structural: no
# gamma, weight or cadence reaches it. A ViT on pixels does not carry the
# Wan patch grid, so this removes the lattice at its origin.
#
# ---------------------------------------------------------------------
# THE THREE MOVING PARTS
#
# 1. FEATURE BASIS.  ladd_feature_source=dinov2. The disc's input latents
#    are VAE-decoded to a 24x32-latent crop (192x256 px, 176x240 after
#    the border trim -- the SAME geometry the pixel critic and the style
#    loss use, so all three measure texture at one scale), DINOv2-ViT-S/14
#    is run on them at NEAR-NATIVE resolution (176x240 -> 182x238, a
#    1.03x round to the patch multiple, NOT ``PretrainedPixelDisc``'s
#    pad-to-square-and-resize-to-518, which is a 2.2x magnification and
#    would rescale the very texture being measured), and its four
#    evenly-spaced block taps feed the EXISTING CCM / CSM / LADDDiscHead
#    stack. The WAN projector is never called; ``ladd_pix_wan_projector_
#    calls`` must read 0.
#
# 2. GENERATOR ROUTE = DIRECT DECODE GRADIENT (2026-08-26).
#    The generator's adversarial gradient flows BACKWARD THROUGH THE VAE
#    DECODE into the DINO critic: gen -> latent chunk -> crop -> decode
#    (graph-on, checkpointed) -> DINOv2 -> CCM/CSM/heads -> RpGAN G-term.
#    ladd_disc_loss_weight=1.0 turns the LADD gen-side block back on;
#    surrogate_critic_enabled=false removes the distilled critic.
#
#    THIS IS THE USER'S FIRST CHOICE, not a deviation. Commissioning
#    words: "check if you have enough memory for gradients and if not
#    just use the surrogate gradient method we already make use of."
#    Memory does allow it, and the surrogate fallback was independently
#    measured BROKEN -- it delivered EXACTLY ZERO gradient to all 825
#    trainable generator parameters (825/825 graph-reachable, total norm
#    0.0; grad cosine to the teacher 0.0097 against a documented healthy
#    +0.986). See analysis/gan_tuning/PIXEL_FEATURE_SOURCE.md 7c.
#
#    SUPERSEDED TEXT (kept so the diff against pixdino_* is readable):
#    surrogate_teacher_backbone=ladd_pixel makes the teacher THIS disc
#    (``LADDDiscriminator.score_pixels``), the distiller fits the latent
#    critic to its value AND its input gradient, and
#    ``generator_surrogate_loss`` serves that critic to the generator
#    every step. Proven-consumed path: surrogate_consumed=1,
#    pix_g_applied=1.
#
# 3. TEACHER CADENCE.  Unlike the SAM2 surrogate arm -- which got ~19
#    head updates in 115 steps and never separated, voiding its result --
#    this teacher's heads AND encoder are updated by the gt_vs_fake
#    D-update at gan_updates_per_step=5 per step. No separate head
#    optimizer, no surrogate_sam2_* key is consulted.
#
# ---------------------------------------------------------------------
# READ THESE KEYS OR THE RUN PROVES NOTHING (all new):
#   ladd_pix_built                 1
#   ladd_pix_disc_forwards         >0 and RISING
#   ladd_pix_src_forwards          >0 (the ENCODER's own counter)
#   ladd_pix_wan_projector_calls   *** MUST BE 0 ***
#   ladd_pix_grid_h/_w             13 x 17 (the DINO token grid)
#   ladd_pix_encoder_trainable     0 for arm A, 1 for arm B
#   ladd_pix_encoder_trainable_params  0 for A, ~22.06M for B
#   ladd_pix_decode_grad           *** MUST BE NON-ZERO ON THIS ARM ***
#       SUPERSEDED TEXT was "0 (the generator pays no decode grad)".
#       That belonged to the SURROGATE arms (pixdino_*), where the
#       generator never backprops through a decode. THIS arm is the
#       DIRECT decode-gradient route, so a non-zero value is exactly
#       what proves the route is live. Measured 204 on all three
#       completed 200-step arms (frozen x2, online x1), against
#       decode_nograd 525. A ZERO HERE WOULD BE THE FAILURE.
#   ladd_pix_decode_nograd         >0
#   ladd_pix_teacher_calls         0 on pixdirect (surrogate is OFF).
#       The ">0 (score_pixels served the surrogate)" expectation is
#       surrogate-only; measured 0 on all three completed arms.
#   ladd_pix_jitter_y/_x           MOVING step to step
#   surrogate_distill_ran          1
#   surrogate_consumed             1
#   pix_g_applied                  1
#   surrogate_grad_probe_site_is_chunk  1
#   surrogate_grad_norm_unweighted      >0  (NOT 0.0 -- see below)
#   surrogate_grad_ratio_unweighted     the calibration read
#   surrogate_grad_severed_from_chunk   *** MUST BE ABSENT ***
#
# The grad probe anchors on the STUDENT CHUNK, not on "the last
# requires_grad generator parameter". The parameter site read 0.0
# EXACTLY, 13/13 samples, on the live w2gram arm while the shared base
# gradient read a healthy 0.032-0.209: autograd returned a non-None
# ALL-ZERO vector because the probed parameter was graph-reachable but
# OFF-PATH (auxiliary heads register after the DiT and are fed by losses
# that never pass through the predicted latent chunk). The student chunk
# is on-path by construction. A zero there would be real.
#
# ---------------------------------------------------------------------
# JUDGE ON TEXTURE, NOT ON GAN HEALTH. Today's variance study puts the
# run-to-run noise floor at 50 % on gan_dmd_grad_ratio, 76 % on r1 and
# 119 % on gan_cos -- no effect on those is claimable at n=1. The
# credible readouts are paired texture measurements on TEXTURED crops
# (tree crowns, road -- NOT sky), each arm against ITS OWN clean_x_real
# decode:
#     analysis/sharpness/fold2d.py       P=16 dotfrac   GT 0.13
#     analysis/sharpness/grid_texture.py mod-8 fold     GT 1.42
#     analysis/sharpness/radial_spec.py  HF content     two-sided
# AND REJECT THE DEGENERATE WIN: if the fold comes down while
# high-frequency content drops BELOW the dataset, that is the ``of``
# smoothing failure, not texture. Matching is two-sided.
# =====================================================================
# ARM B -- BACKBONE ONLINE. The DINOv2 encoder trains as part of the
# disc, from the same gt_vs_fake D loss, in the same optimizer step.
# ASKS: does letting the feature basis adapt to OUR data beat the fixed
# prior? This is the pretrained-encoder form of One-Forcing divergence 1
# (their disc IS the trainable backbone; ours throttles the adversarial
# gradient into it).
#
# WHAT "ONLINE" MEANS HERE, CONCRETELY, AND WHY
# --------------------------------------------
# FULL backbone (all 12 blocks + patch embed, 22.06M params), at 0.1x
# the disc learning rate = 2e-6, in its own AdamW param group. Stated
# alternatives and why they were not taken:
#   * last-N blocks only -- would confound "does adaptation help" with
#     "which depth adapts", and the taps are evenly spaced over ALL the
#     depth (blocks 2/5/8/11), so freezing the early blocks freezes two
#     of the four feature maps the disc actually reads. That is a
#     different experiment.
#   * adapter+norm only -- that is nearly arm A with extra steps.
#   * full backbone at the FULL disc lr (2e-5) -- rejected. The reason
#     to pick DINOv2 over a from-scratch conv is its pretrained basis,
#     and a 21M ViT under an adversarial loss on a moving generator can
#     discard that basis inside tens of steps. B would then lose to A
#     for a reason that has nothing to do with the hypothesis, and we
#     would have measured "adversarial fine-tuning is unstable" instead
#     of "adaptation helps".
#
# THE KICK ACCOUNTING (done deliberately, not inherited)
# -----------------------------------------------------
# The fake-score backbone's neutral scale is 1/(n_modes x
# gan_updates_per_step) because those 10 backwards ACCUMULATE into one
# .grad buffer that one optimizer step then applies -- the scale
# normalises a SUM.
#
# That arithmetic does NOT transfer here, and assuming it did would be
# a 5x error. This encoder sits inside r3gan_optimizer, and the
# positional D-update calls ``r3gan_optimizer.zero_grad(set_to_none=
# True)`` at the top of EVERY one of its 5 iterations and ``.step()``
# at the bottom. So the encoder takes 5 SEPARATE optimizer steps per
# training step, each on ONE update's gradient -- nothing accumulates
# and there is no sum to normalise. One pair mode is enabled
# (gt_vs_fake only), so n_modes = 1 as well.
#
# The right control is therefore the LEARNING RATE, not a grad scale,
# and the quantity to hold is the per-STEP parameter displacement:
# 5 steps x lr, against the heads' own 5 steps x lr. 0.1 buys the
# encoder a tenth of the heads' per-step travel. Under Adam the update
# is ~lr-sized regardless of gradient magnitude, so this is a real 10x
# and not an approximation.
#
# Two consequences to watch, both new keys:
#   * ladd_pix_encoder_trainable_params must read ~22.06M (0 = the flag
#     did not take, which would silently make B a copy of A);
#   * the launch echo prints n_encoder / n_head_adapter / encoder_lr /
#     head_lr, so the param-group split is checkable at boot rather
#     than inferred from the config.
# =====================================================================
# =====================================================================
# ARM E -- "STRONG": MAXIMUM ADVERSARIAL DOSE.
#
# *** THIS IS NOT A SINGLE-VARIABLE ABLATION AND MUST NEVER BE CITED AS
# *** ONE. Three knobs move at once, on purpose. It answers exactly one
# *** question -- "can we move texture AT ALL?" -- and if it wins, the
# *** attribution among the three is UNKNOWN and needs its own arms.
# Commissioning words: "pixdirect online is decisively better than frozen
# but are they both set quite weak? I want more texture improvement, they
# are only giving a bit."
#
# Fork of sbatch/pixdirect_online.sbatch. Changed:
#     ladd_pixel_encoder_lr_scale   0.1 -> 0.4   (same as arm C)
#     ladd_pixel_crops_per_row      1   -> 2     (NOT 3 -- see MEMORY)
#     ladd_pixel_lat_frames         2   -> 3
#     gan_grad_telemetry_every      25  -> 1     (common to all 3 arms)
# Unchanged and deliberately so: gan_loss_weight=1.0, gan_lr=2e-5,
# gan_updates_per_step=5, dfake_gen_update_ratio=5,
# ladd_pixel_encoder_trainable=true, crop_rows/cols, border, jitter.
#
# ---------------------------------------------------------------------
# THE MEASURED BASELINE THIS IMPROVES ON (branch A, counter-derived):
#   * REACH = 8.2 % of the chunk's latent cells per G-update. A 24x32
#     crop is 768 of a 60x104 = 6240-cell latent frame = 12.3 %, at ONE
#     location per step shared by every row, over lat_frames=2 of
#     num_frame_per_block=3 -> 12.3 % x 2/3 = 8.2 %.
#   * The G-term is live on only 36 of 201 iterations (17.9 %) --
#     dfake_gen_update_ratio=5 plus warmup. 900 D-updates against 36
#     adversarial G-updates is 25:1.
#   * online d_loss median 0.5640 (n=8, steps >= 50) against ln2=0.6931.
#     The critic is near chance. frozen is 0.6490, nearer still.
# The critic being near chance is the reason the texture signal is weak:
# a discriminator at chance emits almost no gradient, so the reach and
# the cadence above are multiplying something close to zero.
#
# ---------------------------------------------------------------------
# NEW REACH, computed the same way so the improvement is a NUMBER.
# The K crops are drawn INDEPENDENTLY from the same uniform (model/
# ladd_disc.py:1691-1699 -- K separate randint draws), so coverage is a
# UNION, not a sum. Adding them would be a real over-claim:
#     coverage(K) = 1 - (1 - 0.123)^K
#     K=1: 12.3 %      K=2: 23.1 %      K=3: 32.5 %
# Times the temporal share lat_frames / num_frame_per_block:
#     baseline  K=1, L=2:  12.3 % x 2/3 =  8.2 %   (1.0x)
#     THIS ARM  K=2, L=3:  23.1 % x 3/3 = 23.1 %   (2.8x)
#     rejected  K=3, L=3:  32.5 % x 3/3 = 32.5 %   (4.0x)
# So this arm nearly TRIPLES the reach. 2.8x, not the 3x a naive
# crops_per_row=3 sum would have claimed and not the 4x K=3 would buy.
#
# WHY lat_frames=3 SPECIFICALLY. num_frame_per_block=3, and the disc
# takes ``lat = x_noisy[:, F_lat - L:]`` (ladd_disc.py:1691) -- the
# NEWEST L frames. At L=2 the OLDEST latent frame of every chunk is
# never shown to the discriminator, in any step, ever: it is
# permanently unsupervised by the adversarial loss. L=3 is exactly the
# whole chunk, so this closes the hole rather than merely widening it,
# and L=4 would be pointless (the clamp
# ``L = min(lat_frames, F_lat)`` would cut it back to 3 anyway).
#
# ---------------------------------------------------------------------
# MEMORY -- THE RISK ON THIS ARM. Worked, not asserted.
#
# THE ANCHOR, and a CORRECTION to ONBOARDING_PIXDIRECT.md section 2.
# From job 6135660 (pixdirect_frozen, K=1 L=2, 32 steps, 8 ranks):
#     persistent alloc          38.31 - 42.38 GB
#     TRUE per-step peak        53.35 GB   (ranks 1-7)
# The doc's "peak 42.38 GB" is RANK 0 ONLY, and rank 0 reports
# peak == alloc on every line (38.40/38.40, 42.38/42.38) because the
# [ROLL] block calls torch.cuda.reset_peak_memory_stats() every step
# (trainer/causal_action_forcing_train.py:19073). It understates the
# real peak by ~11 GB. Budget against 53.35 GB, not 42.38.
#     headroom to 95 GB         41.65 GB
#     transient above residency 53.35 - 38.40 = 14.95 GB (conservative)
#
# WORK MULTIPLIERS (model/ladd_disc.py:1687-1730):
#   crops_t = [B*K, L, C, 24, 32] and the closure at trainer:7010-7038
#   hands the WHOLE tensor to _vae_decode_grad in ONE shot, so
#     decode work  ~ B * K * L
#   imgs = [B*K*kf, 3, 176, 240] with kf = frames_per_crop = 2 FIXED, so
#     DINO work    ~ B * K   only -- lat_frames does NOT enlarge the ViT
#                                   batch, only the decode.
#       K=3, L=3:  decode x4.5,  DINO x3
#       K=2, L=3:  decode x3.0,  DINO x2
#       K=2, L=2:  decode x2.0,  DINO x2
#
# METHOD 1 -- SOUND PESSIMISTIC BOUND. Assume the ENTIRE 14.95 GB
# transient is pixel-path and all of it scales with the decode
# multiplier (it does not, but this cannot under-estimate):
#     peak <= 38.40 + M x 14.95
#     K=3,L=3 (M=4.5):  38.40 + 67.3 = 105.7 GB   OOM on a 95 GB card
#     K=2,L=3 (M=3.0):  38.40 + 44.9 =  83.3 GB
#     K=2,L=2 (M=2.0):  38.40 + 29.9 =  68.3 GB
#
# WHY METHOD 1 IS KNOWN LOOSE, with evidence. The non-pixel parent
# gansig_gtvf_dmd_carnpure (job 6130135) -- same DMD/GAN structure,
# latent-tap disc, NO VAE decode anywhere in the disc -- runs rank-0
# step_peak 52.23-59.80 GB against this arm's 38.40-49.79 on the same
# rank-0 measurement. A run with no decode at all carries a LARGER
# transient than this one. So the DMD/disc backward, not the decode,
# dominates the 14.95 GB, and the part that actually scales with K*L is
# a minority of it.
#
# METHOD 2 -- SIZE THE DECODE COMPONENT DIRECTLY. The output tensor is
# negligible: [B*K, 4L, 3, 192, 256] fp32 at B=3,K=1,L=2 is 14.2 MB.
# The cost is the VAE decoder's internal activations, and
# _vae_decode_grad wraps them in torch.utils.checkpoint(use_reentrant=
# False); its docstring puts the saving at "~5-10 GB" at this geometry,
# i.e. 5-10 GB is the UN-checkpointed decode cost at K=1,L=2 and the
# checkpointed one is a fraction of that. Treating the whole 5-10 GB as
# the scaling component (again over-counting, since checkpointing
# removes most of it):
#     K=2,L=3 (M=3.0): 15-30 GB, of which 5-10 is already inside the
#                      measured 53.35 -> ADD 10-20 GB -> ~63-73 GB
#     K=3,L=3 (M=4.5): ADD 17.5-35 GB                 -> ~71-88 GB
#
# THE TWO METHODS BRACKET IT:
#     K=2, L=3:  63 - 83 GB
#     K=3, L=3:  71 - 106 GB
#
# DECISION: SHIP crops_per_row=2, NOT 3.
# K=3 exceeds the 75 GB gate under BOTH methods' upper ends and has a
# credible path past 95 GB (OOM) under method 1. K=2,L=3's CENTRAL
# estimate clears 75 GB and only its pessimistic tail does not, and that
# tail comes from the method proven above to over-count. The reach paid
# for this is 23.1 % instead of 32.5 % -- 2.8x the baseline instead of
# 4.0x. That is still the largest reach increase any arm in this
# campaign has had, and an OOM at step 25 buys 0x and burns a holder.
#
# RETREAT LADDER if it OOMs anyway, in order:
#   1. crops_per_row 2 -> 1, keep lat_frames=3  (M=1.5, ~+3-8 GB over
#      the online arm) -- still closes the unsupervised-frame hole.
#   2. lat_frames 3 -> 2 as well = back to the online footprint exactly.
# Both are env-free one-line edits to this file.
#
# *** NOT AVAILABLE AS A MITIGATION: ladd_pixel_decode_batch. ***
# It is INERT on this route and lowering it would be a silent no-op of
# precisely the kind ONBOARDING_PIXDIRECT.md section 9 warns about.
# trainer:7004 writes it into disc.pixel_cfg and the boot echo prints
# it, but the ONLY reader of pixel_cfg is model/ladd_disc.py:1687-1730,
# which consults crop_rows, crop_cols, crops_per_row, lat_frames,
# border and frames_per_crop -- and never "decode_batch". The batched
# decoder that DOES honour a batch size (_pix_decode_crops_grad,
# trainer:6879-6915) belongs to the pixel-critic/surrogate path and
# reads a DIFFERENT key, pix_decode_batch (trainer:6652); the surrogate
# is OFF here. Left at its inherited 4 rather than tuned. Making it
# real is a SOURCE change and is out of scope for this file -- reported,
# not fixed.
#
# CACHE. ladd_pixel_decode_cache=true and the last frozen arm ran
# hit/miss = 340/185, so ~65 % of decode calls are served from the cache
# and only the 185 misses pay the x3.0. That is what makes this
# affordable; it is also why the memory peak (set by a single worst-case
# miss) scales more than the wall clock does.
#
# WALL CLOCK -- flagged, not silently patched. Decode x3.0 and DINO x2
# on a 33 s/step budget where the pixel path is a minority of the step
# suggests ~35-45 s/step. At 45 s, 200 steps = 150 min against the
# 02:10:00 (130 min) wall. checkpoint_interval=50 means partial results
# survive, and judging needs medians past step 50 which is reached
# either way -- but if a full 200 is wanted, launch with MAXSTEPS=150
# (the ${MAXSTEPS:-200} env shape is preserved). MAXSTEPS is NOT changed
# in this file, because the brief said change nothing else.
#
# ---------------------------------------------------------------------
# PROVE THE FLAGS FIRED -- per knob, with the value to expect.
#
# BOOT ECHOES (stderr; under the holder pattern they land in
# logs/holdersmoke_pixdirect_strong_h<HOLDER>.log via 2>&1):
#   [LADD-PIXFEAT] decode callback INSTALLED: crop=24x32 lat
#       (=192x256 px) crops_per_row=2 lat_frames=3 frames_per_crop=2
#       border=8 decode_batch=4 cache=True
#   [LADD-PIXFEAT] encoder param group: ... lr_scale=0.4
#       encoder_lr=8e-06 head_lr=2e-05
#   [PIXDIRECT-KNOBS] ... enc_lr_scale=0.4
#
# COUNTERS (wandb summary, NOT the log -- section 9). Baseline column is
# the last frozen arm, measured:
#
#   key                    frozen    THIS ARM      why
#   ---------------------  --------  ------------  ----------------------
#   ladd_pix_images          3558    ~7116  (x2)   B*K*kf, K 1->2, kf=2
#   ladd_pix_disc_forwards    729    ~729   (=)    counts CALLS, not size
#   ladd_pix_src_forwards     729    ~729   (=)    one encoder call per
#                                                  disc forward; MUST
#                                                  equal disc_forwards
#   ladd_pix_decode_grad      204    ~204   (=)    call count
#   ladd_pix_decode_nograd    525    ~525   (=)    call count
#   ladd_pix_grid_h/_w      13/17    13/17  (=)    *** KEY CHECK ***
#   ladd_pix_wan_projector_
#       calls                   0    0             *** MUST BE 0 ***
#   cache hit/miss        340/185    ~340/185 (=)  pattern is structural
#
# THE DISCRIMINATING CHECK IS THE RATIO, not the raw count, because
# forwards vary with ride/reset scheduling:
#     images / disc_forwards :  frozen 3558/729 = 4.88
#                               THIS ARM must read  ~9.76
# A ratio still near 4.88 means crops_per_row did NOT take and this arm
# is a replicate of pixdirect_onlinelr. STOP and do not report a result.
#
# grid_h/_w staying 13x17 is the check that the RIGHT knob moved: K and
# L change the BATCH, never the per-image geometry (crop_rows, crop_cols
# and border are untouched). A grid that MOVES means crop geometry
# changed, i.e. the wrong knob moved.
#
# *** TELEMETRY GAP, stated rather than papered over: lat_frames has NO
# *** dedicated counter. images scales with K and kf but not with L;
# *** disc_forwards / src_forwards / decode_grad / decode_nograd are all
# *** call counts. So the ONLY direct proof that lat_frames=3 took is the
# *** boot echo above, with step-time and peak memory as indirect
# *** corroboration. Worse, ladd_disc.py:1690 clamps
# *** L = max(1, min(lat_frames, F_lat)) SILENTLY -- with
# *** num_frame_per_block=3 we expect F_lat=3 and L=3 exactly, but if the
# *** disc's input ever arrived with F_lat=2 the clamp would be invisible
# *** in every counter. Flagged, not fixed (source change, out of scope).
#
# ---------------------------------------------------------------------
# SUCCESS CRITERION. Texture, two-sided, on TEXTURED crops (tree crown,
# road -- NEVER sky), each arm against ITS OWN clean_x_real decode:
#     analysis/sharpness/fold2d.py        P=16 dotfrac   GT 0.13
#     analysis/sharpness/grid_texture.py  mod-8 A8y      GT 1.42
#     analysis/sharpness/radial_spec.py   HF content     two-sided
# This arm is judged on whether texture MOVES MATERIALLY toward those
# numbers, not on whether it moves in a clean attributable way.
# Secondary: d_loss median past step 50 should fall from 0.5640 toward
# the 0.25-0.55 band -- a critic with 2.8x the reach and 4x the encoder
# lr should be further from chance than one at 0.5640.
#
# WHAT WOULD FALSIFY IT.
#   * NULL: texture unchanged within the paired-crop spread AND d_loss
#     median still ~0.564. Then the adversarial dose is not the limiting
#     factor at all -- neither reach, nor temporal coverage, nor encoder
#     lr -- and the bottleneck is the 25:1 D:G cadence or the route
#     itself. That is a genuinely useful negative and closes three axes
#     at once, which is the whole reason for combining them here.
#   * D-WINS: d_loss <= 0.135 on CONSECUTIVE logged steps -- the w003
#     failure. Three knobs were pushed toward a stronger D, so this is
#     the most likely way this arm breaks. It is a FAILURE, not a win:
#     the generator stops receiving usable gradient. Do NOT respond by
#     cutting gan_loss_weight ("w003 is bad, w1 is better"); back off
#     crops_per_row or ladd_pixel_encoder_lr_scale instead.
#   * DEGENERATE WIN: fold comes down while radial HF drops BELOW the
#     dataset -- the ``of`` smoothing failure. Matching is two-sided.
#   * OOM: see the retreat ladder above.
#
# JUDGE ON MEDIANS PAST STEP 50 (ONBOARDING_PIXDIRECT.md section 7).
# Never a single step, never the final step alone, and never an n=1
# claim on gan_dmd_grad_ratio (50 % CV), r1 (76 %), d_real or gan_cos
# (95-119 %). d_loss CV is 14-17 %, so the effect must clear ~15 %.
#
# ---------------------------------------------------------------------
# ALSO CHANGED, COMMON TO ALL THREE NEW ARMS (not an arm variable):
#   gan_grad_telemetry_every  ${TELEM:-25} -> ${TELEM:-1}
#       The parent inherits =1 in the baseline block and then overrides
#       it to ${TELEM:-25} AFTER $DEXTRA; last-wins means 25 actually
#       applied, which is why a 200-step run yielded only ~2 usable
#       ratio/cos readings. The grandparent (gansig_gtvf_dmd_carnpure)
#       ran the same probe at 1 with no reported cost. Restored to 1 by
#       the researcher's approval. The ${TELEM:-N} env-override shape is
#       kept so a run can dial it back without editing the file.
# =====================================================================

# ===================== PIXDINO OVERRIDES =============================
# Everything above this banner is gansig_gtvf_dmd_carnpure verbatim.
#
# ladd_feature_source=dinov2
#     THE change. Swaps the disc's feature basis from WAN taps of
#     LATENTS to DINOv2 on decoded PIXELS.
#
# ladd_fake_backbone_trainable=false   (was true)
#     FORCED, and the resolver raises if you set it otherwise. That flag
#     releases the adversarial freeze on the FAKE-SCORE backbone, and a
#     pixel-source disc never touches that backbone -- so leaving it true
#     would be silently inert. One-Forcing divergence 1 is therefore not
#     under test on this arm; ``ladd_fake_backbone_grad_scale`` is inert
#     for the same reason and is left at its inherited value rather than
#     deleted, so the diff against the parent stays readable.
#
# ladd_disc_loss_weight=1.0   (was 0.0 on the surrogate arms)
#     Turns the LADD GENERATOR term back ON -- this is the whole route
#     change. gen_gan_weight is multiplied by it, so 0.0 skipped the
#     gen-side block entirely and 1.0 restores it. The D-update is
#     unaffected either way.
#
#     SUPERSEDED NOTE from the surrogate arms:
#     Kills the LADD GENERATOR term (gen_gan_weight is multiplied by it,
#     and the gen-side block is gated on gen_gan_weight > 0). The
#     D-update is untouched -- n_disc_updates depends on disc_skipped,
#     not on the gen weight -- so the disc still trains on gt_vs_fake at
#     five updates per step. This is what makes the generator's ONLY
#     adversarial route the surrogate, per the researcher's design.
#
# ladd_disc_force_clean=true  (inherited, and now REQUIRED)
#     The model-side validator raises without it: the pixel path decodes
#     the disc's input, and a latent re-noised at disc_t decodes to
#     noise. Independently the measured-better setting (TEXTURE_REVIEW
#     2.2, two single-variable pairs).
#
# ladd_pixel_grid_jitter=8
#     THE ANTI-LATTICE TERM, and the load-bearing one. fold2d measures
#     PHASE-LOCKED structure; an encoder whose grid phase moves every
#     step cannot accumulate a phase-locked artefact. Applied as an
#     offset INSIDE the border trim that is discarded anyway, so it
#     costs nothing and pads nothing. Set 0 for the control.
#
# surrogate_teacher_backbone=ladd_pixel
#     The teacher IS this disc. Distilling the critic the adversarial
#     loop actually trains -- rather than a parallel copy with its own
#     pairing and its own warm-up -- is the whole premise.
#
# pix_teacher_refresh_every=2
#     The distiller runs the teacher (a graph-on decode + DINO forward +
#     input-gradient) every N steps and REPLAYS cached targets in
#     between; the critic serves the generator every step regardless.
#     That is the entire affordability mechanism. 2, not the config's 4:
#     the teacher here moves fast (5 D-updates/step on both heads and,
#     in arm B, the encoder), so stale targets go stale quickly. Read
#     train/surrogate_n_teacher_refresh to confirm the cadence took --
#     never a raw forward count, which is 2x under checkpointing.
#
# pix_r1_gamma=0.0
#     Explicitly OFF, and explicitly SET: _pix_resolve_cfg raises on an
#     unset gamma, and the pixel critic's own R1 has nothing to
#     regularise here (there is no pixel PatchGAN in this arm). The LADD
#     disc's own R1 is unchanged at ladd_r1_gamma=10.0.
#
# pix_gan_weight
#     NO DEFAULT, by repo discipline, and NOT inheritable from the LADD
#     arms (different domain, reduction, loss form and critic). CALIBRATE:
#     launch the smoke at PIXW=0.0, read train/surrogate_grad_ratio_
#     unweighted (the share the term WOULD have at weight 1.0 -- at
#     weight 0 the applied term has no gradient at all, so only that key
#     can answer), then solve PIXW = 0.10 / r for the 5-20 % band.
# =====================================================================
# =====================================================================
# 2026-08-26 RULING -- gt_vs_fake POSITIVES ARE NEVER NOISED.
#
# The researcher ruled that the gt_vs_fake discriminator must never see
# CARN-noised GT in its positive rows. This is now the TRAINER DEFAULT:
# the pair-mode gate in _ladd_run_pair_mode bypasses the forward noiser
# whenever pair_mode == "gt_vs_fake", and the escape hatch
# ``forward_noiser_allow_gt_vs_fake`` defaults to FALSE. Clean positives
# are what happens when nobody sets anything.
#
# CONSEQUENCE FOR THIS FILE: the ``forward_noiser_apply_gt_former=true``
# line below is now INERT ON THE gt_vs_fake REAL SIDE. It is retained
# deliberately, as documentation of intent and because it still governs
# gt_transition (which is the CARN model's territory and explicitly
# outside the ruling). It is NOT dead config; it is scoped config.
#
# PROVE THE BYPASS FROM COUNTERS, never from this comment:
#     train/fn_gtvf_noise_skipped_gt   MUST rise on this arm
#     train/fn_gtvf_noise_applied_gt   MUST stay 0.0
#   NOTE ON THE KEY NAME: _compute_ladd_losses suffixes EVERY per-mode
#   key with the pair mode (trainer:11986, suffix _gt / _gtxn / _adj).
#   The emitted key is therefore fn_gtvf_noise_skipped_GT, not the bare
#   name. Grepping the bare name returns ABSENT, which is exactly the
#   'absent key reads as zero/broken' hazard this counter exists to
#   prevent. Same convention as r3gan_d_loss_gt.
# Both are wandb-only -- read wandb/wandb/<run>/files/wandb-summary.json,
# not the log. In the job log:
#     grep "FN-DECOUPLED-APP"  -> must NOT show pair_mode=gt_vs_fake
#     grep "FN-GTVF-CLEAN"     -> must show "BYPASS site=..."
#
# PROVENANCE: arms launched BEFORE this change have NOISED positives and
# must not be pooled with post-change arms. Pre-change runs on holders
# 6140643/6140644/6143212/6143213 are affected.
# =====================================================================
#SBATCH --job-name=pxd-dstrg
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:10:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
DARM=${DARM:-pixdirect_strong}
ENC_TRAIN=${ENC_TRAIN:-true}
ENC_LR_SCALE=${ENC_LR_SCALE:-0.4}
# PIXW is NOT consulted on the direct route: there is no pixel-critic
# G-term and no surrogate. The generator's adversarial weight is
# gan_loss_weight x ladd_disc_loss_weight, both inherited from the
# parent arm, so no un-calibrated knob is introduced.
PIXW=${PIXW:-0.0}
ODE_CKPT=${ODE_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt}
# WALL-CLOCK ARITHMETIC (2-node holder, measured on the CARN arms this is
# forked from -- v6na step21->step71 = 1629 s / 50 = 32.6 s/step; v6seam
# = 32.0 s/step; 6allon = 33.1 s/step, all INCLUDING the sample_interval=15
# video renders).  Stripping CARN can only make it faster, so budget the
# CARN number: 200 steps x 33 s = 6600 s = 110 min, + ~5 min boot
# (srun overlay + model load + first ride) = ~115 min ~ the 2 h target.
MAXSTEPS=${MAXSTEPS:-200}
DEXTRA=${DEXTRA:-}
CONFIG=configs/action_forcing_phase3_dmd.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/dmd10k_${DARM}
WANDB_SAVE_DIR=wandb
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export ARRWM_ACTION_ENCODER=pca_raw
export TMPDIR=/tmp
unset LOCALDIR APPTAINER_CACHEDIR
export ARRWM_MANIFEST_PICKLE=/scratch/u6ex/as1748.u6ex/ARRWM/analysis/.dmd_weunz_manifest_min69.pt
mkdir -p "$LOGDIR" logs
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn NCCL_DEBUG=WARN NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
echo "[ARRWM-PROVENANCE] HEAD=$(git rev-parse HEAD) tree=$(git stash create 2>/dev/null || echo CLEAN) srcmd5=$(cat model/dmd_action_forcing.py model/anti_collapse.py model/ladd_disc.py pipeline/action_forcing_training.py trainer/causal_action_forcing_train.py | md5sum | cut -d' ' -f1)"
git status --porcelain | head -20
echo "GANSIG arm=$DARM ckpt=$ODE_CKPT steps=$MAXSTEPS nodes=$SLURM_NNODES $(date)"
echo "[PIXDIRECT-KNOBS] arm=pixdirect_strong feature_source=dinov2 enc_trainable=$ENC_TRAIN enc_lr_scale=$ENC_LR_SCALE pix_gan_weight=$PIXW g_route=DIRECT_DECODE_GRADIENT ladd_disc_loss_weight=1.0 surrogate=OFF jitter=8 ARM_VARIABLE=MULTI(enc_lr0.4+crops2+latf3)_NOT_AN_ABLATION telem=1"
echo "[GANSIG-KNOBS] arm=gansig_gtvf_dmd force_clean=true gtvf_only=true carn=decoupled critic=frozen src=dmd gan_loss_weight=1.0 gan_lr=2e-5 gan_updates_per_step=5 gan_warmup_steps=25 ladd_r1_gamma=10.0 ladd_r1_every_n_steps=1 ladd_r1_num_samples=3 ladd_fake_backbone_grad_scale=0.1 ladd_gt_transition_enabled=false ladd_gt_vs_fake_enabled=TRUE ladd_gt_vs_fake_all_pairs=false ladd_gt_vs_fake_wide_real=false ladd_adjacent_chunks_enabled=false ladd_fake_sample_source=DMD ladd_readout=ladd(default) ladd_disc_micro_batch_groups=8"

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
    max_steps=$MAXSTEPS \
    checkpoint_interval=${CKPT_EVERY:-50} \
    keep_last_n_checkpoints=4 \
    save_full_checkpoint=false \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_weunz \
    holdout_zarr_list=configs/holdout_rides.txt \
    holdout_eval_root=/projects/u6ex/fbots/frodobots_encoded_weu_holdout \
    holdout_eval_mode=cycle \
    sample_7chunk_enabled=true \
    max_ride_frames=900 \
    max_ride_frames_random=false \
    seed=1234 \
    ema_start_step=30 \
    action_teacher_mode="off" \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    action_dims=[0,1] \
    action_critic_dims=[0,1] \
    denoising_step_list=[1000,625,357.142857,208.333333] \
    max_rolls_per_ride=4 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=21 \
    num_workers=0 \
    sample_fps=16 \
    boundary_vae_roundtrip=false \
    local_attn_size_schedule=[[0,21]] \
    infinity_rope=true \
    seed_prefill_mode=real \
    kv_cache_seed_headroom=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=0.0 \
    dmd_context_clean_frames=9 \
    streaming_chunk_size=18 \
    dmd_loss_weight=1.0 \
    dmd_loss_start_step=0 \
    dmd_loss_warmup_steps=20 \
    dmd_normalization_enabled=true \
    dmd_mae_gate_enabled=true \
    dmd_mae_gate_min_weight=1.0 \
    dmd_mae_gate_r_full=2.0 \
    dmd_mae_gate_ema=0.0 \
    dmd_mae_gate_exponent=0.75 \
    dmd_sample_at_rungs=true \
    dmd_real_traj_enabled=false \
    dmd_real_traj_max_steps=6 \
    dmd_debug_step=208 \
    stat_anchor_loss_weight=${STAT_ANCHOR:-1.0} \
    stat_anchor_mode=gt_window \
    stat_anchor_use_clean_match_offset=true \
    stat_anchor_match_k=2 \
    stat_anchor_M2_short_weight=0.1 \
    stat_anchor_M2_long_weight=0.0 \
    stat_anchor_TV_short_weight=0.1 \
    stat_anchor_TV_long_weight=0.0 \
    stat_anchor_M1_short_weight=6e-8 \
    stat_anchor_M1_long_weight=0.0 \
    stat_anchor_MEAN_short_weight=0.9 \
    stat_anchor_MEAN_long_weight=0.0 \
    stat_anchor_SOS_short_weight=0.0 \
    stat_anchor_SOS_long_weight=0.0 \
    stat_anchor_STD_short_weight=0.0 \
    stat_anchor_STD_long_weight=0.0 \
    stat_anchor_rel_tol_short=0.0 \
    stat_anchor_rel_tol_long=0.0 \
    dmd_ar_head_commit=student \
    dmd_asymmetric_scoring_enabled=false \
    dmd_42f_enabled=true \
    dmd_42f_2chunk=false \
    dmd_42f_seed_last=false \
    dmd_42f_gt_after_chunks=${D42_AFTER:-1} \
    dmd_42f_rand_sup_slot=false \
    dmd_42f_allsup=false \
    dmd_42f_gt_anchor=true \
    dmd_42f_num_chunks=${D42_NS:-4} \
    dmd_42f_fix_clean_counterpart=false \
    dmd_42f_clean_drift_enabled=false \
    dmd_42f_clean_match_enabled=true \
    dmd_42f_clean_match_max_drift_frames=12 \
    dmd_42f_clean_match_min_improve=0.15 \
    dmd_42f_clean_match_forward=false \
    dmd_42f_clean_match_forward_only=true \
    dmd_42f_rolling_sup_new=true \
    dmd_42f_allroll_student_ctx=true \
    dmd_only_first_chunk_per_ride=false \
    num_chunks_roll_forward=3 \
    streaming_min_new_frame=9 \
    streaming_max_length=60 \
    motion_start_threshold=5.0 \
    fake_score_lora_enabled=false \
    real_teacher_train_online=false \
    dmd_frozen_teacher_pass_enabled=false \
    real_score_ema_weight=0.0 \
    aux_teacher_loss_weight=0.0 \
    aux_real_clean_x_source=gt \
    aux_noisy_from_raw_gt=true \
    aux_clean_x_random_window=false \
    use_8bit_adam=true \
    gen_gradient_checkpointing=true \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    strict_ode_load=true \
    ode_generator_checkpoint=$ODE_CKPT \
    v14_teacher_checkpoint=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    student_phase_lora_enabled=false \
    forward_noiser_enabled=false \
    carn_recurse=false \
    forward_noiser_apply_gt_former=false \
    carn_seam_affine_lambda=0.0 \
    ladd_gt_match_frames=0 \
    flash_dmd_enabled=true \
    flash_dmd_gan_t=60 \
    gan_enabled=true \
    gan_backbone=ladd_teacher_feat \
    gan_gate_couple_enabled=false \
    gan_max_grad_norm=10.0 \
    gan_disc_start_step=20 \
    gan_critic_warmup_steps=20 \
    gan_warmup_shape=linear \
    ladd_r1_normalize_tokens=true \
    ladd_r1_unified_cadence=false \
    ladd_r1_num_samples=3 \
    ladd_r1_sigma=0.01 \
    ladd_feature_blocks=[0,2,4,8,29] \
    ladd_proj_dim=256 \
    ladd_cmap_dim=0 \
    ladd_use_prompt_cond=false \
    ladd_scalar_output=false \
    ladd_freeze_projector_mixing=true \
    ladd_use_csm=true \
    ladd_disc_head_kernel=3 \
    ladd_disc_loss_weight=1.0 \
    ladd_diff_aug_policy=flip,translation \
    ladd_gt_transition_enabled=true \
    ladd_gt_transition_match=false \
    ladd_gt_vs_fake_enabled=true \
    ladd_adjacent_chunks_enabled=false \
    ladd_wavelet_hf_enabled=false \
    ladd_wavelet_hf_augment=false \
    ladd_feature_source=fake \
    ladd_fake_backbone_trainable=true \
    ladd_fake_backbone_grad_norm_every=1 \
    ladd_defer_disc_update=true \
    ladd_disc_micro_batch_groups=8 \
    ladd_gen_guidance_micro_batch_groups=8 \
    lr=1e-5 \
    fake_lr=1e-5 \
    streaming_fake_updates_per_gen=4 \
    fake_score_ema_weight=0.0 \
    fake_score_init_from_teacher=true \
    dmd_grad_target_norm=1.0 \
    gan_grad_telemetry_every=1 \
    gan_real_diversity_log=true \
    texture_tripwire_every=1 \
    gan_loss_weight=1.0 \
    gan_lr=2e-5 \
    gan_updates_per_step=5 \
    gan_warmup_steps=25 \
    ladd_disc_force_clean=true \
    ladd_r1_gamma=10.0 \
    ladd_r1_every_n_steps=1 \
    ladd_fake_backbone_grad_scale=0.1 \
    ladd_gt_vs_fake_weight=1.0 \
    ladd_gt_transition_weight=1.0 \
    ladd_fake_sample_source=dmd \
    $DEXTRA \
    ladd_gt_transition_enabled=false \
    ladd_gt_vs_fake_enabled=true \
    ladd_fake_backbone_grad_scale=0.2 \
    action_critic_aux_enabled=true \
    action_critic_freeze=true \
    action_critic_checkpoint=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    action_critic_dims=[0,1] \
    action_teacher_mode=off \
    generator_action_z_guidance_weight=0.3 \
    z_guidance_warmup_steps=50 \
    ladd_gside_checkpoint_recover=true \
    ladd_action_origin_track_slice=true \
    ladd_action_origin_log=true \
    gan_health_show_r1_fire_rate=true \
    gan_health_show_hold_step=true \
    ladd_match_distance_log=true \
    ladd_backbone_grad_scale_check=true \
    forward_noiser_enabled=true \
    forward_noiser_apply_decoupled=true \
    forward_noiser_apply_gt_former=true \
    forward_noiser_chain_levels=true \
    forward_noiser_former_mode=weak \
    forward_noiser_reverse=true \
    forward_noiser_loss_mode=teacher_feat \
    forward_noiser_apply_gt_level=1 \
    carn_recurse=false \
    ladd_feature_source=dinov2 \
    gan_grad_telemetry_every=${TELEM:-1} \
    ladd_fake_backbone_trainable=false \
    ladd_disc_loss_weight=1.0 \
    ladd_pixel_crop_rows=24 \
    ladd_pixel_crop_cols=32 \
    ladd_pixel_crops_per_row=2 \
    ladd_pixel_lat_frames=3 \
    ladd_pixel_frames_per_crop=2 \
    ladd_pixel_border_trim=8 \
    ladd_pixel_decode_batch=4 \
    ladd_pixel_decode_cache=true \
    ladd_pixel_grid_jitter=8 \
    ladd_pixel_dino_variant=dinov2_vits14 \
    ladd_pixel_dino_n_taps=4 \
    ladd_pixel_encoder_trainable=${ENC_TRAIN} \
    ladd_pixel_encoder_lr_scale=${ENC_LR_SCALE} \
    surrogate_critic_enabled=false \
    run_name=dmd10k_${DARM}_j${SLURM_JOB_ID}

echo "GANSIG $DARM completed $(date)"
