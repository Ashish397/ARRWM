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
# 2. GENERATOR ROUTE.  The generator gets NO decode gradient. The LADD
#    G-term is switched off (ladd_disc_loss_weight=0.0, which zeroes
#    ``gen_gan_weight`` and skips the gen-side block outright) and the
#    adversarial gradient arrives instead through WP-SURROGATE:
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
#   ladd_pix_decode_grad           0 (the generator pays no decode grad)
#   ladd_pix_decode_nograd         >0
#   ladd_pix_teacher_calls         >0 (score_pixels served the surrogate)
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
# ladd_disc_loss_weight=0.0
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
#
# ---------------------------------------------------------------------
# 2026-08-26 -- THE SURROGATE ZERO-GRADIENT FIX (PIXEL_FEATURE_SOURCE.md
# 7c). Read this before launching; the arm was UNLAUNCHABLE without it.
#
# 7c measured, live: surrogate_param_grad_norm_unweighted = 0 over 825/825
# graph-reachable generator params (DMD base 9.53), surrogate_check_cos_sim
# 0.0097 against a documented-healthy ~0.986, rel_err 0.99997. The critic
# was reachable and returning a NON-zero value (surrogate_g_value 0.0465)
# while its INPUT-GRADIENT field -- the only thing the generator consumes,
# since generator_surrogate_loss is -dense.mean() -- was nothing.
#
# ROOT CAUSE, measured on CPU and arithmetically exact: the critic was
# never trained. _maybe_run_surrogate_distillation is called from
# _streaming_train_one_chunk, which runs ONLY on generator iterations
# (self.step % dfake_gen_update_ratio == 0, and dfake_gen_update_ratio=5
# here), and it early-returns below gan_disc_start_step=20. A 90-step run
# therefore calls it on steps 20,25,...,85 = 14 calls, of which
# should_refresh (step % pix_teacher_refresh_every=2) fires on 7 -- which
# is EXACTLY the surrogate_n_teacher_refresh=7 that run reported. So the
# latent critic got FOURTEEN gradient steps in the entire run. At
# MAXSTEPS=200 it gets 36.
#
# CPU measurement of what it needs. Held-out surrogate_check_cos_sim vs
# NUMBER OF CRITIC GRADIENT STEPS (n=1 per point, one seed -- read the
# shape, not the third digit):
#
#   critic steps          7      14      30      60     120     250     500
#   realistic teacher   0.380   0.572   0.676   0.758   0.803   0.826   0.841
#   white-noise teacher 0.003  -0.003  -0.001   0.020   0.034   0.312   0.707
#
# Production sat at 14. The white-noise row at 14 reads -0.003, which is
# production's 0.0097 to within noise -- and it still reaches 0.707 by
# 500. The failure is a STEP BUDGET in both regimes. The shipped
# hyperparameters are FINE; the number of steps was not.
#
#   surrogate_distill_substeps=12   <-- THE FIX, and the load-bearing one
#       12 critic gradient steps per distillation call, replaying the
#       CACHED teacher targets. The teacher (a graph-on decode + DINO
#       forward + input gradient) still fires only on the
#       pix_teacher_refresh_every cadence, so the expensive half of the
#       arm is unchanged; only the critic's own forward/double-backward
#       on 4 small crops repeats. 36 calls x 12 = 432 critic steps, past
#       the ~120 the sweep says is enough.
#       Single-variable CPU pair, same call count and IDENTICAL teacher
#       refresh count: substeps=1 -> held-out cos 0.2567; substeps=8 ->
#       0.7640 (n=1 each, small-crop harness, so read the SIGN and the
#       order of magnitude, not the third digit).
#
#   surrogate_critic_head_init_std=0.02   <-- mechanism, not measured win
#       value(z) = w.h(z)+b with h the mean-pooled token feature, so the
#       generator-facing gradient cg = d(value)/dz = J_h(z)^T w is EXACTLY
#       linear in the head weight. The shipped head is ZERO-init
#       (model/latent_texture_critic.py), so at step 0 cg == 0 and
#       dL_grad/d(trunk) == 0 -- measured 0.000e+00, not "small" --
#       and the trunk is the only part that can rotate its Jacobian
#       toward the teacher's field. With the trunk fixed the reachable
#       gradient subspace is only d_model-dimensional: the closed-form
#       least-squares ceiling on the cosine measures 0.140 against the
#       random-subspace value sqrt(d_model/n_z) = 0.144.
#       HONEST CAVEAT: once the step budget is fixed this flag's effect on
#       the end metric is WITHIN NOISE at n=1 (120-step CPU sweep, patch
#       teacher: 0.7910 with vs 0.7873 without). It is set because it
#       removes an exactly-zero-gradient pathology at no cost, NOT because
#       a measurement demanded it. Set it to 0.0 to reproduce the shipped
#       initialisation byte for byte.
#
#       Note the ONLY place in the repo where the Sobolev term is proven
#       to work -- testing/test_latent_texture_critic.py::_fit -- silently
#       does nn.init.normal_(critic.head.weight, std=0.02) right after
#       construction and runs at lr=3e-3 with default Adam betas,
#       refresh_every=1 and max_grad_norm=1.0. Five divergences from
#       production, so the passing test never exercised the shipped path.
#       That is where the "documented healthy ~0.986" came from.
#
# WHAT DID *NOT* EXPLAIN IT (measured, so it does not get re-litigated):
#   * value-loss dominance: value_loss_weight -> 0 changes the 120-step
#     held-out cosine from 0.8063 to 0.8079. FALSIFIED.
#   * grad_loss_normalize: L_grad pinned at ~1.000 is the CONSEQUENCE of a
#     near-zero critic field (L_grad = 1 - 2*r*cos + r^2 with r ~ 0.006),
#     not a flattening of the signal.
#   * the second-order backward through the hand-rolled attention: it
#     propagates. Trunk grads from L_grad are non-zero and scale linearly
#     in |w| the moment |w| > 0.
#   * teacher refresh cadence: pix_teacher_refresh_every is UNCHANGED by
#     this fix and the two CPU arms above hold it identical.
#
# PROVE IT FIRED FROM THE COUNTER, NEVER FROM THIS FILE:
#   train/surrogate_distill_substeps      12   (the configured value)
#   train/surrogate_n_distill_substeps    critic gradient steps TAKEN
#   train/surrogate_substeps_achieved     ***  MUST READ ~12.0, NOT 1.0 ***
#         = n_distill_substeps / (n_teacher_refresh + n_replay). This is
#         the key that distinguishes "flag is in the config" from "flag
#         was honoured". OmegaConf.from_dotlist accepts unknown keys
#         SILENTLY; both keys above have a read site in
#         model/latent_texture_critic.py::build_from_config.
#   train/surrogate_check_cos_sim         *** THE VERDICT ***, emitted every
#         surrogate_grad_check_every=25 steps. It read 0.0097 on the broken
#         run. Anything still under ~0.1 means the fix did not take and the
#         arm proves nothing -- do NOT read texture off such a run.
#   train/surrogate_check_rel_err         was 0.99997; must come well off 1.
#   train/surrogate_grad_norm_unweighted  must be > 0.
#
# STILL UNPROVEN, and it needs GPU: whether the LIVE teacher's gradient
# field is spatially smooth (like PatchTeacher, where the shipped recipe
# reaches cos 0.81) or high-frequency. Against a WHITE-noise teacher the
# same sweep tops out at the d_model ceiling ~0.14 and only a larger
# surrogate_critic_lr helps (0.0692 -> 0.5836 at lr=1e-3). lr is left at
# the shipped 2.0e-4 because raising it is unmeasured at 432 steps and one
# CPU config collapsed at 3e-3 (cos -0.005, mag-ratio 4e-4). If
# surrogate_check_cos_sim stalls near 0.14 on GPU, that ceiling -- not the
# step budget -- is the live regime, and surrogate_critic_lr is the next
# knob, not pix_gan_weight.
#
# FALSIFIABLE PREDICTION, read both keys on the SAME run. 7c's own table
# says surrogate_param_n_reached_sur = 825 ("not severed, not off-path"),
# which contradicts the `base_unreachable` reading quoted in
# docs/ONBOARDING_PIXDIRECT.md 3. The parameter-space probe is the later
# and stronger measurement, so the working hypothesis is that the exact
# zero was a genuinely tiny gradient flushing to zero. If after this fix
# surrogate_check_cos_sim > 0.6 AND surrogate_grad_norm_unweighted is
# STILL exactly 0, then there is a SECOND defect and it is a plumbing one
# -- stop and fix that before reading any texture off the run.
# ---------------------------------------------------------------------
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
#SBATCH --job-name=pxd-online
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
DARM=${DARM:-pixdino_online}
ENC_TRAIN=${ENC_TRAIN:-true}
ENC_LR_SCALE=${ENC_LR_SCALE:-0.1}
# pix_gan_weight has NO default -- see the banner.
if [ -z "${PIXW}" ]; then
  echo "[PIXDINO] FATAL: PIXW (pix_gan_weight) unset. "\
       "Run the PIXW=0.0 probe first, read "\
       "train/surrogate_grad_ratio_unweighted, then set "\
       "PIXW = 0.10 / r. There is deliberately no default." >&2
  exit 1
fi
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
echo "[PIXDINO-KNOBS] arm=pixdino_online feature_source=dinov2 enc_trainable=$ENC_TRAIN enc_lr_scale=$ENC_LR_SCALE pix_gan_weight=$PIXW g_route=surrogate ladd_disc_loss_weight=0.0 teacher=ladd_pixel jitter=8 substeps=${SUBSTEPS:-12} head_init_std=${HEADSTD:-0.02}"
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
    gan_grad_telemetry_every=${TELEM:-25} \
    ladd_fake_backbone_trainable=false \
    ladd_disc_loss_weight=0.0 \
    ladd_pixel_crop_rows=24 \
    ladd_pixel_crop_cols=32 \
    ladd_pixel_crops_per_row=1 \
    ladd_pixel_lat_frames=2 \
    ladd_pixel_frames_per_crop=2 \
    ladd_pixel_border_trim=8 \
    ladd_pixel_decode_batch=4 \
    ladd_pixel_decode_cache=true \
    ladd_pixel_grid_jitter=8 \
    ladd_pixel_dino_variant=dinov2_vits14 \
    ladd_pixel_dino_n_taps=4 \
    ladd_pixel_encoder_trainable=${ENC_TRAIN} \
    ladd_pixel_encoder_lr_scale=${ENC_LR_SCALE} \
    surrogate_critic_enabled=true \
    surrogate_teacher_backbone=ladd_pixel \
    surrogate_critic_lr=2.0e-4 \
    surrogate_value_loss_weight=1.0 \
    surrogate_grad_loss_weight=1.0 \
    surrogate_grad_loss_normalize=true \
    surrogate_grad_check_every=25 \
    surrogate_distill_substeps=${SUBSTEPS:-12} \
    surrogate_critic_head_init_std=${HEADSTD:-0.02} \
    pix_teacher_refresh_every=${TEACHER_EVERY:-2} \
    pix_crop_lat=[24,32] \
    pix_crops_per_step=4 \
    pix_frames_per_crop=2 \
    pix_lat_frames_per_crop=2 \
    pix_decode_border_trim=8 \
    pix_decode_batch=4 \
    pix_real_pool_windows=2048 \
    pix_real_pool_refresh=8 \
    pix_real_pool_warm_updates=32 \
    pix_gan_weight=${PIXW} \
    pix_r1_gamma=0.0 \
    run_name=dmd10k_${DARM}_j${SLURM_JOB_ID}

echo "GANSIG $DARM completed $(date)"
