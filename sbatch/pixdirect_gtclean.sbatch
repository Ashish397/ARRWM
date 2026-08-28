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
# ARM D -- CLEAN GT POSITIVES.  Fork of sbatch/pixdirect_online.sbatch.
# ONE knob moves:
#     forward_noiser_apply_gt_former   true -> false   (LAST occurrence)
# The encoder settings are UNCHANGED from pixdirect_online
# (ladd_pixel_encoder_trainable=true, ladd_pixel_encoder_lr_scale=0.1)
# ON PURPOSE: this arm's comparison partner is the LIVE pixdirect_online
# run, so it inherits that arm's encoder exactly and gets an n=2 baseline
# for free. It is NOT to be compared against pixdirect_onlinelr (arm C),
# which moves a different knob.
#
# ---------------------------------------------------------------------
# WHY. This is not a hypothesis about the noiser, it is a COUNTER-PROVEN
# fact about what the discriminator's POSITIVE class currently contains.
# The live arm prints, from trainer/causal_action_forcing_train.py:13806:
#
#   [FN-DECOUPLED-APP] ACTIVE: pair_mode=gt_vs_fake cpp=1 mode=weak
#   chain=True uncond=... n=3 levels=[0,1,1] maxlvl=...
#
# Read that literally. n=3 rows of "real" per pair batch, per-row CARN
# levels [0, 1, 1]. Level 0 keeps the ORIGINAL clean GT; levels >= 1 are
# run through the forward noiser. So TWO OF THREE rows the disc is being
# taught to call "real" are CARN-NOISED GT, and only ONE is genuine
# dataset footage.
#
# And the noising is MOMENT-PRESERVING by construction -- the block
# restores the per-channel mean and then rescales to the original
# mean-absolute value (trainer/causal_action_forcing_train.py:13790-13798)
# before handing the tensor back as real_chunks_det. Mean and scale
# survive; what does NOT survive is TEXTURE STRUCTURE. Which is precisely
# and only the quantity this entire workstream exists to match:
# "it's not just sharpness, our main objective has to be style shift to
# the dataset style."
#
# So on a texture-objective arm, 2/3 of the target the disc defines is
# not the dataset's texture. That is worth one single-variable arm, and
# the researcher approved running it as one.
#
# ---------------------------------------------------------------------
# THE LAST-WINS TRAP -- READ THIS BEFORE EDITING.
# Overrides are LAST-WINS, and the parent sets this key TWICE, on purpose:
#     ~line 424 (BEFORE $DEXTRA, baseline block):  =false
#     ~line 502 (AFTER  $DEXTRA, arm block):       =true   <-- WINS
# The parent therefore RESOLVES TO TRUE, and the early "false" is inert
# and always was. This arm changes the LATE occurrence -- the one after
# $DEXTRA -- to false, and leaves the early one exactly as it was. The
# winning line for this arm is the post-$DEXTRA one, and it reads false.
# Resolved by parsing the full override list, not by eyeballing.
#
# ---------------------------------------------------------------------
# WHAT THIS DOES *NOT* DO -- scope, precisely.
#   * forward_noiser_enabled stays TRUE and the forward noiser is still
#     TRAINED (forward_noiser_loss_mode=teacher_feat, reverse=true). Only
#     its APPLICATION to the discriminator's positives is removed. This
#     is not "turn CARN off", it is "stop feeding the disc CARN'd reals".
#   * forward_noiser_apply_decoupled stays TRUE but becomes INERT: the
#     decoupled block is gated on apply_gt_former OR apply_gt_both
#     (model/dmd_action_forcing.py:2367-2372; the trainer gate at
#     causal_action_forcing_train.py:13705-13718), and
#     forward_noiser_apply_gt_both is not set anywhere in this arm, so it
#     is False by default. With gt_former=false the whole block is
#     skipped. Left SET rather than deleted so the diff against the
#     parent stays a genuine one-line diff and stays readable.
#   * The parent's own baseline block already documents the historical
#     silent no-op this decoupling was built to fix; that history is
#     unchanged, we are only choosing not to use the capability here.
#
# ---------------------------------------------------------------------
# PROVE THE FLAG FIRED. This is the one case where an ABSENCE is valid
# proof, because the banner is UNCONDITIONAL inside the block it gates:
#   [FN-DECOUPLED-APP] ACTIVE   *** MUST BE ABSENT from stderr ***
# (it prints up to 3 times on the main process, guarded only by
# _fn_decoupled_dbg < 3, so if the block runs at all it is seen). Grep
# the log of THIS job, never the parent's: under the holder pattern
# run_smoke_on_holder.sh redirects 2>&1 into
# logs/holdersmoke_pixdirect_gtclean_h<HOLDER>.log, so stderr lands
# there; under a real sbatch it is the %x_%j.err file.
# Everything else must be unchanged from the live online arm:
#   ladd_pix_encoder_trainable          1
#   ladd_pix_encoder_trainable_params   22,056,192
#   [LADD-PIXFEAT] ... lr_scale=0.1 encoder_lr=2e-06
#   ladd_pix_wan_projector_calls        *** MUST BE 0 ***
#
# ---------------------------------------------------------------------
# WHAT WOULD FALSIFY IT.
#   * NULL: paired texture metrics on TEXTURED crops -- fold2d P=16
#     dotfrac (GT 0.13), grid_texture mod-8 A8y (GT 1.42), radial_spec HF
#     -- unchanged within the paired-crop spread against the live online
#     arm. Then the CARN'd positives were never material to texture and
#     this axis closes. Two-sided: undershooting is as wrong as
#     overshooting, and an HF drop BELOW the dataset is the ``of``
#     smoothing failure, not a win.
#   * D-WINS: cleaner positives make the disc's job EASIER (it no longer
#     has to call two structurally-degraded rows "real"), so d_loss can
#     fall. If the median past step 50 lands at or below the documented
#     0.135 D-wins threshold on consecutive steps, that is a real and
#     informative failure -- the CARN noise was functioning as a disc
#     handicap. Do NOT fix that by cutting gan_loss_weight (see w003);
#     the lever would be gan_lr / gan_updates_per_step.
#   * Healthy target remains the 0.25-0.55 band, judged on MEDIANS PAST
#     STEP 50 -- never a single step, never the final step alone
#     (ONBOARDING_PIXDIRECT.md section 7; d_loss CV is 14-17 %, and
#     ratio/r1/d_real/cos are 50-119 % and not claimable at n=1).
#
# ---------------------------------------------------------------------
# WHAT IS DELIBERATELY NOT TOUCHED.
#   gan_loss_weight=1.0  -- "w003 is bad, w1 is better". Untouched.
#   gan_lr=2e-5, gan_updates_per_step=5 -- untouched.
#   ladd_pixel_encoder_trainable=true, ladd_pixel_encoder_lr_scale=0.1 --
#     deliberately IDENTICAL to the live online arm, so the pairing is
#     against a running baseline rather than against another new arm.
#   real_guidance_scale=0.0, dmd_ar_head_weight untouched (TF head only),
#     clean context never noised.
#
# ---------------------------------------------------------------------
# ALSO CHANGED, COMMON TO BOTH NEW ARMS (not the arm variable):
#   gan_grad_telemetry_every  ${TELEM:-25} -> ${TELEM:-1}
#       The parent inherits =1 in the baseline block and then overrides it
#       to ${TELEM:-25} AFTER $DEXTRA, and last-wins means 25 actually
#       applied -- which is why a 200-step run yielded only ~2 usable
#       ratio/cos readings. The grandparent arm (gansig_gtvf_dmd_carnpure)
#       ran the same probe at 1 with no reported cost. Restored to 1 by
#       the researcher's approval. The ${TELEM:-N} env-override shape is
#       kept so a run can still dial it back without editing the file.
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
# STATUS CHANGE 2026-08-26 -- THIS ARM IS NO LONGER A TREATMENT.
#
# The researcher ruled that the gt_vs_fake discriminator must NEVER see
# noised GT in its positive rows, and that is now the DEFAULT in the
# trainer (see the pair-mode gate in _ladd_run_pair_mode and the
# counters train/fn_gtvf_noise_skipped_gt / _applied). This arm's variable,
# forward_noiser_apply_gt_former=false, is therefore SUBSUMED BY THE
# DEFAULT and no longer changes anything.
#
# WHAT THIS ARM IS NOW: an exact A/A REPLICATE OF pixdirect_online
# (ENC_TRAIN=true, ENC_LR_SCALE=0.1 -- identical, verified). Keep it for
# the run-to-run noise floor, which this campaign is chronically short
# of. Do NOT list it as a treatment in any results table.
#
# THE ONE COMPLETED RUN IS STILL A VALID TREATMENT. The run on holder
# 6140643 was launched BEFORE the trainer change, when this flag really
# did suppress the noiser, so gtclean-vs-online within the PRE-change
# set is a genuine single-variable comparison. Do not pool pre-change
# and post-change runs.
#
# The flag also disabled the noiser for gt_transition, which was
# harmless here only because ladd_gt_transition_enabled=false on this
# arm. It was always a blunt instrument.
#
# For a deliberate noised-positives ablation use the escape hatch
# forward_noiser_allow_gt_vs_fake=true instead, NOT this flag.
# =====================================================================
#SBATCH --job-name=pxd-dgtcl
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
DARM=${DARM:-pixdirect_gtclean}
ENC_TRAIN=${ENC_TRAIN:-true}
ENC_LR_SCALE=${ENC_LR_SCALE:-0.1}
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
echo "[PIXDIRECT-KNOBS] arm=pixdirect_gtclean feature_source=dinov2 enc_trainable=$ENC_TRAIN enc_lr_scale=$ENC_LR_SCALE pix_gan_weight=$PIXW g_route=DIRECT_DECODE_GRADIENT ladd_disc_loss_weight=1.0 surrogate=OFF jitter=8 ARM_VARIABLE=NONE_POST_2026-08-26_AA_REPLICATE_OF_pixdirect_online telem=1"
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
    forward_noiser_apply_gt_former=false \
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
    surrogate_critic_enabled=false \
    run_name=dmd10k_${DARM}_j${SLURM_JOB_ID}

echo "GANSIG $DARM completed $(date)"
