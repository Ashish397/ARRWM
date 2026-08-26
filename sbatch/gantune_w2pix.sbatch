#!/bin/bash
# VARIANT: w2pix  --  gantune_w2 PLUS the FROM-SCRATCH PIXEL-SPACE TEXTURE
# PatchGAN (WP-PIXGAN / B1, docs/TEXTURE_GAN_DESIGN.md).  EXACT copy of
# sbatch/gantune_w2.sbatch (the already-measured w=2.0 tuning point) with
# ONE mechanism added, so the delta vs gantune_w2 is attributable.
#
# WHY THIS ARM EXISTS.  The LADD latent disc STRUCTURALLY CANNOT SEE
# RESOLUTION: it scores transformer features of LATENTS at a fixed t=60,
# projected 1536->256, taps [0,2,4,8,29].  It improves geometry, not
# texture -- confirmed by the researcher's own viewing.  This arm is the
# first one in the campaign whose critic looks at DECODED RGB PIXELS.
#
# WHAT IT IS (read model/pixel_texture_disc.py, not the name): a 661,953-
# param PatchGAN, 3->64->128->256->1, k4/k4/k4/k3, spectral norm on all
# four convs, LeakyReLU(0.2), GroupNorm REMOVED (authorised deviation from
# TEXTURE_GAN_DESIGN §4).  Per-patch logits, no global scalar.  Real and
# fake are BOTH decode(latent) through the same frozen Wan VAE, so the
# decoder's transfer function cancels and the critic cannot reward the
# generator for out-sharpening the decoder.  Fake = the flash t=60 x0
# (pix_finish_grad_enabled=false); real = GT latent crops from the A21
# refresh pool, drawn UNIFORM and band-matched (never nearest-matched).
#
# =====================================================================
# THE FIVE KNOWN ISSUES, AND WHAT THIS FILE DOES ABOUT EACH
# =====================================================================
#
# (a) MEMORY -- expected peak ~71 GB on the 95.00 GiB cards.
#     Baseline: gantune_w2 peaks ~65 GB at GAN-ACTIVE steps at full
#     rolling depth.  (The 46.05/49.18/49.19/49.41/48.84 GB readings at
#     steps 0/5/10/15/20 are EARLY-STEP readings and must not be quoted
#     as the peak -- that mistake is what made run 1 of w2sam look safe.)
#     ADDED COST, itemised:
#       * G-side GRAPH-ON decode (trainer::_pix_decode_crops_grad ->
#         _vae_decode_grad, torch.utils.checkpoint'd).  Sized from the
#         MATCHED-STEP w2sam measurement rather than guessed: w2sam run 1
#         showed +42.4 GB transient at step 20 over gantune_w2 on the same
#         holder, and the re-size (crops 8->4 halves the decode term,
#         sam2_resolution 512->256 cuts the SAM2 term to 1/8) projected
#         ~8 GB.  Solving  D + S = 42.4  and  D/2 + S/8 = 8.0  gives
#         D ~= 7.2 GB (decode, at crops=8) and S ~= 35.2 GB (SAM2 Hiera
#         activations retained for the input-gradient pass).  THE SAM2
#         TERM IS ABSENT FROM THIS ARM.  Only D applies.
#       * pix_decode_batch=2 (YAML ships 4) halves the in-flight
#         checkpoint-recompute sub-batch, so the ~7.2 GB figure is an
#         upper bound; realistic ~4-5 GB.  The RETAINED output is trivial:
#         8 crops x 3 frames x 3 x 176 x 240 fp32 = 12.2 MB.
#       * D-loop decodes are no_grad (TEXTURE_GAN_DESIGN §5.2), serialised
#         in pix_decode_batch sub-batches and freed each iteration:
#         transient < 1 GB, NOT additive across the 5 D-updates.
#       * critic fwd/bwd: 0.66 M params on <= 24 images at 176x240 -- and
#         R1 is a finite difference on PIXELS (no decode), one extra
#         critic forward.  Both sub-GB.
#       * A21 real pool is fp16 ON CPU (4096 x 2 x 16 x 24 x 32 ~ 200 MB
#         host, 0 GB GPU).
#     => 65 + ~5 + ~1 ~= 71 GB, i.e. ~24 GB headroom.  w2sam run 1 died
#     with 0.4 GB of headroom (91.80 GB peak / 94.58 GiB in use) when the
#     sample_interval=15 7-chunk 168-frame render raised the floor at
#     step 36; 24 GB is ~60x that margin.
#     DOWNSCALE RULE, pre-registered: if step_peak exceeds 82 GB BEFORE
#     the first sample render, drop pix_crops_per_step 8 -> 4.  Crops is
#     the ONLY linear memory axis here (docs/WP_PIXGAN.md §9: frame
#     selection happens strictly AFTER decode, so pix_frames_per_crop
#     costs nothing at decode).  Do NOT cut pix_reals_per_fake (A20) or
#     pix_real_pool_windows (A21) -- those are correctness, not budget.
#
# (b) FROM-SCRATCH DISC -- pix_gan_lr=1e-4, NOT the shipped 1e-5.
#     1e-5 is the LADD regime: a projection head on top of a PRETRAINED
#     Wan backbone, where the features already exist and only the mixing
#     is learned.  This critic has no pretrained prior at all -- it must
#     learn real dashcam texture filters from noise while the generator
#     moves under it.  The reference implementations of THIS EXACT
#     architecture (pix2pix; taming-transformers/VQGAN, i.e. the very
#     from-scratch pixel PatchGAN that produced the latent space we train
#     in) run 2e-4.  1e-4 is 10x the shipped value, half the reference,
#     with R1 on as the guard -- see (d).
#     BUDGET: pix_gan_updates_per_step is left NULL so it FOLLOWS
#     gan_updates_per_step=5.  Past gan_disc_start_step=20 a 400-step arm
#     gives ~1,900 D-updates against TEXTURE_GAN_DESIGN §4.1's stated
#     "~180 D-updates to learn texture statistics from nothing" risk --
#     ~10x margin.
#     WARM-UP IS DELIBERATELY NOT CHANGED.  gan_disc_start_step=20 /
#     gan_warmup_steps=25 are SHARED with the LADD disc (_pix_gen_weight
#     anchors its ramp on exactly those two keys), so moving them would
#     also move the LADD arm and destroy comparability with gantune_w2 /
#     w2sam / w2dino / w2cnx.  CONSEQUENCE, stated so nobody reads it as
#     a result: the pixel D gets ZERO updates before the generator first
#     feels it, and the linear ramp over steps 20-45 is its only head
#     start.  That is why the read window starts at step 45, why MAXSTEPS
#     defaults to 400, and why pix_d_loss is read as a TRAJECTORY and
#     never as an endpoint (§5.5).
#
# (c) CROP STATISTICS + THE BLOCKY/PIXELATED LIABILITY.
#     A PatchGAN on small crops is a local texture-statistics matcher
#     with no global anchor, and the cheapest way to satisfy a WEAK local
#     critic is high-contrast hard-edged structure -- i.e. exactly the
#     blocky/pixelated failure the researcher has ruled out.  This
#     codebase has already been bitten by that class: see
#     analysis/vertical_banding/, an 8px stippled/halftone imprint
#     phase-locked to the VAE decoder's own upsampling grid, which
#     fn_spatial_fix_README.md shows CANNOT be removed by any rms-
#     preserving latent-space reshaping (76.8% of the perturbation energy
#     lives in the same low-fy band as the function) -- it exists only
#     after the decoder, so it can only be measured after the decoder.
#     WHAT PREVENTS IT HERE (four things, none of them wishful):
#       1. CROP SIZE IS NOT REDUCED.  pix_crop_lat=[24,32] = 192x256 px,
#          176x240 after the 8-px border trim = 40% of frame height.  A
#          smaller crop is precisely the locally-plausible-globally-wrong
#          regime; it is not a memory knob here (crops COUNT is).
#       2. RECEPTIVE FIELD ~70 px against a 176-px crop -- local, but not
#          tiny.  The old objection that "GroupNorm makes gradient
#          support span the whole image" does NOT apply: the shipped
#          module has use_norm=False, and the measured pairwise patch
#          alpha is -0.85..-1.05 (mean -0.947), i.e. patch logits really
#          are near-independent local questions.
#       3. R1 AT A CALIBRATED, NON-ZERO GAMMA -- see (d).  R1 bounds
#          ||grad_x D|| on REAL pixels, which is exactly a penalty on the
#          discriminator becoming a sharp local edge/contrast detector --
#          the D shape that rewards block structure.
#       4. SAMPLE COUNT.  crops=8 x frames=3 = 24 fake images per
#          D-update (~144 effective non-overlapping RF tiles vs the
#          original draft's ~24), and A24 band matching (pix_band_count=3)
#          means a fake ROAD crop is never scored against a real SKY
#          crop.  Variance is the thing that lets a degenerate local
#          optimum look acceptable.
#     HOW IT IS DETECTED (two instruments, both live in this arm):
#       * texture_tripwire_every=1 is INHERITED from gantune_w2 and this
#         arm additionally sets texture_tripwire_gt_ref=true, which adds
#         train/tripwire_gt_* and train/tripwire_ratio_* -- the SAME
#         anisotropic battery computed on the frame-matched GT and
#         reported as a RATIO.  That is what turns a one-sided "more HF"
#         reading into a two-sided DISTANCE TO THE DATASET.
#         ABORT RULE, pre-registered:
#             tripwire_ratio_hf_power        > 1.5 and still rising, OR
#             tripwire_ratio_angular_entropy < 0.90, OR
#             tripwire_ratio_hv_anisotropy   outside 0.7 .. 1.4
#         = the arm is buying high-frequency energy with GRID STRUCTURE
#         (blocky/aliased content is axis-aligned, so angular entropy
#         goes DOWN and anisotropy walks away from 1 -- the measured
#         C_late signature).  Kill it; do not "tune through" it.
#       * POST-HOC on the sample-interval rollout mp4s:
#         analysis/vertical_banding/phase_fold.py measures A8y/A16y --
#         8px / 16px phase-locked grid amplitude in 8-bit luma units,
#         directly.  Reference readings already on disk in
#         phase_fold_results.json: clean GT A8y=1.00 / A16y=1.57; this
#         family's own pred_fake A8y~0.63 / A16y~0.70.  A8y or A16y
#         climbing ABOVE the GT reading while the eye reports "sharper"
#         is the halftone failure, not a win.
#     RECORDED GAP: ladd_diff_aug_policy=flip,translation is inherited
#     and applies to the LADD path only.  The pixel path has NO DiffAug
#     (ADA/augmentation is scope-frozen out of B2 until the A22 held-out
#     test MEASURES D-overfit).  If D overfits, that is the escalation.
#
# (d) pix_r1_gamma = 200 (NOT 0.0, and it cannot be left unset).
#     The YAML ships pix_r1_gamma: null and the trainer's
#     _pix_resolve_r1_gamma REFUSES BOTH null AND 1.0, on purpose.
#     1.0 is MEASURED INERT on the shipped norm-free critic:
#         R1/d_loss = 3.66e-06 .. 7.65e-06  against d_loss = 2 ln 2
#     over 10 matched inits at P=660 (the real post-trim 176x240 crop).
#     Pinning 1.0 launches the arm with R1 effectively OFF while
#     train/pix_r1_rate reads a confident 1.00 -- the exact silent-
#     failure shape this campaign keeps hitting.
#     The MEASURED gamma that puts R1 at 0.1% of d_loss on the shipped
#     norm-free critic at P=660 is 131..273.  200 is the middle of that
#     band, and pix_crop_lat is UNCHANGED from the calibration crop, so
#     the number transfers.
#     WHY NON-ZERO AND WHY SUBSTANTIAL: 0.0 is a legal, explicit value
#     and it is the RIGHT value for the surrogate arms (w2sam/w2dino/
#     w2cnx), whose generator-facing critic is the LATENT one and whose
#     pixel disc is off.  It is the WRONG value here.  On the LADD side
#     R1 mattered enormously -- ladd_r1_gamma=10.0 with every_n=1 is now
#     the codebase default precisely because the unregularised disc
#     saturated -- and this critic is from scratch, unregularised, and
#     structurally able to find the block shortcut (see (c)).
#     STILL CALIBRATE, DO NOT ASSUME.  train/pix_r1_grad_sq(_mean) is
#     logged PRE-gamma and train/pix_r1_grad_share gives R1's POST-gamma
#     share of the D-side gradient.  Read them on the FIRST logged row
#     past step 20:
#         pix_r1_grad_share ~ 1e-3          -> gamma is right, leave it
#         pix_r1_grad_share < 1e-4          -> R1 inert; raise toward 273+
#         pix_r1_grad_share > 1e-1          -> R1 dominating; drop to 131
#     Also assert train/pix_r1_rate == 1.00 on the first row (§5.4): a
#     value below 0.99 is a BUILD BUG, not a tuning outcome.
#     pix_r1_num_samples is left at the YAML default (null = all reals)
#     deliberately: the §5.4 cost argument for subsampling was written
#     against a full-depth Wan forward; here R1 is one extra forward of a
#     0.66 M-param conv on already-decoded pixels, so subsampling would
#     buy nothing and only add variance.
#
# (e) THE CONFOUND -- what a null result would and would NOT tell us.
#     This arm changes TWO things at once, and a third by inheritance:
#       * WHERE the disc looks: decoded RGB pixels vs t=60 transformer
#         features of latents;
#       * WHAT the disc is: a from-scratch 0.66 M-param conv PatchGAN vs
#         a 1536->256 projection of a pretrained Wan backbone;
#       * and the LADD disc STAYS ON (gan_enabled=true,
#         ladd_gt_transition_enabled=true), because the controlled base
#         of this sweep is gantune_w2 and the already-queued surrogate
#         arms made the same choice.  Cross-arm comparability wins; the
#         cost is that the pixel term is NOT measured in isolation.
#         (This is a RECORDED DEVIATION from TEXTURE_GAN_DESIGN §1,
#         which parks the transition critic in every arm of its ladder.
#         The isolated variant is one env var away and is NOT the default:
#           DEXTRA='ladd_gt_transition_enabled=false gan_loss_weight=0.0'
#         -- DEXTRA is spliced AFTER every key below, so last-wins works
#         in our favour.)
#     A POSITIVE result IS attributable to "adding decoded-pixel
#     adversarial feedback", which is the B2 hypothesis and is what we
#     want to learn.
#     A NULL RESULT IS NOT A FALSIFICATION OF PIXEL-SPACE TEXTURE
#     SUPERVISION.  It is equally consistent with at least four causes
#     that this arm alone cannot separate:
#       (i)   the critic is undertrained -- d_loss ~ ln 2 reads
#             IDENTICALLY for "undertrained" and "design is wrong"
#             (§4.1).  The instrument that makes it decidable is the §8.1
#             positive control, which is DEFAULT OFF here (see below);
#       (ii)  pix_gan_weight is wrong.  It is UNCALIBRATED on the first
#             launch of this file -- that is why the file's DEFAULT MODE
#             IS THE WEIGHT-FREE PROBE (see PIXW below).  There is NO
#             GAN-side gradient cap to catch an over-large weight: A4
#             gan_grad_target_norm was DELETED 2026-08-23;
#       (iii) the from-scratch architecture is too weak -- in which case
#             the escalation is a PRETRAINED pixel backbone (GAN_REDESIGN
#             B5 = exactly the w2sam/w2dino/w2cnx surrogate arms), NOT
#             abandoning pixel-space supervision;
#       (iv)  the fake source is the flash t=60 tensor, not the rendered
#             ladder endpoint (pix_finish_grad_enabled=false).  A23 is
#             OVERTURNED so this is an experimental variable, not a gate,
#             but it is a live alternative explanation:
#             pix_finish_grad_enabled=true is the follow-up.
#     So: read a null as "escalate to the pretrained-backbone arms",
#     never as "pixels do not work".
#
# =====================================================================
# TWO LAUNCH MODES -- the file defaults to the PROBE, on purpose
# =====================================================================
#   PROBE   (default):  PIXW=0.0  MAXSTEPS=120
#     pix_gan_weight=0.0 is LEGAL and is exactly the §12 weight-free
#     probe.  The G-term contributes NO gradient, but the full graph-on
#     decode still runs and train/pix_gan_grad_ratio publishes the
#     UNWEIGHTED ratio -- the share the term WOULD have at weight 1.0,
#     which is the only way to get that number (at weight 0 every
#     WEIGHTED readout is exactly 0.0 and settles nothing).
#     The probe is NOT a wasted holder: it also trains the pixel D for
#     the whole run, so it returns the pix_d_loss trajectory, the A20
#     per-update invariants, the A21 support curve, the band histogram,
#     the R1 share, AND the real memory peak.
#     Then solve   pix_gan_weight = 0.10 / r_probe   for the 5-20% band.
#   GRADED:  PIXW=<w>  MAXSTEPS=400   (e.g.  PIXW=12 sbatch ...)
#     Bracket x1/3, x1, x3 around the solved weight.
#   Optional, and only AFTER an offline amplitude calibration
#   (`python -m model.pixel_texture_disc calibrate` / sweep_c1_amplitude):
#     PIX_POSCTL=25  turns on the §8.1 positive control, which is what
#     makes pix_d_loss ~ ln 2 DECIDABLE.  It is OFF by default because an
#     UNcalibrated control returns a confident WRONG verdict, which is
#     worse than no verdict.
#
# LAUNCH GATES NOT MET, recorded rather than hidden:
#   * A22 held-out D generalisation probe (disc_holdout_probe_every) is
#     left OFF.  GAN_REDESIGN A22 is REOPENED with the verdict "I would
#     not trust this probe to gate a real experiment": its leak half
#     FAILS OPEN (an unset/misspelled holdout_eval_root yields an empty
#     reserved set and the exact green signature while checking nothing
#     -- note this arm DOES set holdout_eval_root, so that half would at
#     least be live), and its memorisation half is measured-unusable
#     (train_from_observed=True builds the train pool from ONE ride at
#     the first fire; Monte-Carlo on a critic that memorises NOTHING
#     returns MEMORISING 8.5%, INVERTED 15%, spurious real_auc>0.7 in
#     30% of runs).  Enabling it would add a confident wrong verdict,
#     not an instrument.
#   * §8.1 positive control: OFF pending the offline calibration above.
#   Both are prerequisites for the FOLLOW-UP arm, not for this read.
#
# PROVE IT (the run must show ALL of these, or the arm is a no-op):
#   stdout : "[ActionForcing] WP-PIXGAN resolved config (DERIVED values"
#   wandb  : gen/train/pix_cfg_gan_pixel_texture_enabled == 1
#            gen/train/pix_d_loss                        present
#            gen/train/pix_dupdate_total                 MONOTONE
#            gen/train/pix_r1_rate                       == 1.00
#            gen/train/pix_gan_grad_ratio                present (unweighted)
#            gen/train/pix_real_repeat_frac              == 0.0   (A20)
#            gen/train/pix_real_images == pix_fake_images         (A20)
#            gen/train/pix_a21_support_ok                -> 1 after warm-up
#            gen/train/pix_fake_source_ladder            == 0.0
#            gen/train/pix_g_applied == 1 and pix_g_via_gen_gan_loss == 1
#            gen/train/tripwire_ratio_hf_power           present
#            ABSENT: any pix_r2_* key (would mean the build drifted)
#   In GRADED mode additionally: gen/train/pix_g_loss != 0 and the A7
#   gan_dmd_grad_ratio moving off the measured gantune_w2 trajectory.
#
# =====================================================================
# GAN-TUNING SWEEP (2026-08-25).  analysis/gan_tuning/PLAN.md
#
# SIMPLE BASE = smk_v6's effective (last-wins) override set with the CARN
# forward-noiser STRIPPED, i.e. exactly the parent recipe
# sbatch/train_dmd3_roll.sbatch (rolling streaming DMD3 + LADD GAN on
# ladd_teacher_feat + flash-DMD) but carrying v6's AUDITED hyperparameters
# (fake_score_ema 0.0, lr/fake_lr 1e-5, r1_gamma 1.0 not 1e6, wavelet OFF).
# CARN off  ->  forward_noiser_enabled=false, forward_noiser_apply_gt_former
# =false, carn_recurse=false, carn_seam_affine_lambda=0.0  (the exact three
# CARN keys train_dmd3_roll.sbatch sets).  Everything else held FIXED across
# the sweep, so the GAN knobs are the ONLY moving part.
#
# HELD FIXED IN EVERY ARM (do not vary):
#   ODE_CKPT   = ode14e_pilot run3 step-400   (NOT dmd3kl-200)
#   real_guidance_scale      = 0.0
#   carn_seam_affine_lambda  = 0.0
#   ladd_wavelet_hf_enabled  = false   (wavelet HF kills the gt_transition GAN)
#   stat_anchor_loss_weight  = 1.0     (0.0 collapses the run to black ~step 61)
#   gan_backbone=ladd_teacher_feat, ladd_feature_source=fake,
#   ladd_fake_backbone_trainable=true, ladd_gt_transition_match=false
#   dfake_gen_update_ratio=5, streaming_fake_updates_per_gen=4
#
# JUDGE BY (grep GAN-HEALTH):
#   d_loss   healthy 0.25-0.55.  ->0.005 = disc saturated (it has won).
#   gap = d_real - d_fake   healthy ~1-3.  >5 = saturated.
#   ratio (train/gan_dmd_grad_ratio, WEIGHTED gan grad / DMD grad at the last
#          trainable generator param)   target band 0.05-0.20, held ACROSS the
#          run.  <0.01 = GAN irrelevant.  >1 = GAN dominating DMD.
#   r1     must be NON-ZERO once ladd_r1_every_n_steps=1 (in the current
#          unmatched arms it reads exactly 0 at every logged step).
#   disc_skipped must be 0 after gan_disc_start_step.
#   backbone_grad_norm  spikes to 50-500 = the disc host is being kicked.
#   + the sample videos every 15 steps (7-chunk 168f rollout) = collapse check.
# =====================================================================
#SBATCH --job-name=gt-w2pix
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=06:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
DARM=${DARM:-gantune_w2pix}
ODE_CKPT=${ODE_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt}
# WALL-CLOCK ARITHMETIC (2-node holder, measured on the CARN arms this is
# forked from -- v6na step21->step71 = 1629 s / 50 = 32.6 s/step; v6seam
# = 32.0 s/step; 6allon = 33.1 s/step, all INCLUDING the sample_interval=15
# video renders).  Stripping CARN can only make it faster, so budget the
# CARN number: 200 steps x 33 s = 6600 s = 110 min, + ~5 min boot
# (srun overlay + model load + first ride) = ~115 min ~ the 2 h target.
#
# W2PIX WALL-CLOCK.  gantune_w2 measured 33.50 s/step (steps 21->161,
# incl. the sample_interval=15 renders).  This arm adds, per step:
#   * 5 D-updates x (8 fake + 8 real) no_grad crop decodes = 80 crop
#     decodes = 20 sub-batch decoder calls at pix_decode_batch=2 -> 40;
#   * 1 graph-on crop decode of 8 crops (+ ~30% checkpoint recompute in
#     the backward);
#   * 40 PatchGAN fwd/bwd (0.66 M params -- noise);
#   * A21 pool refresh zarr loads.  THIS IS THE THROUGHPUT RISK, not the
#     memory one: num_workers=0 in this recipe, so the loader is on the
#     critical path.  pix_real_pool_warm_updates is raised 64 -> 128
#     below, which halves the warm-up load rate from 64 to 32 window
#     loads per D-update (160/step for ~26 steps) and leaves steady state
#     at pix_real_pool_refresh=8 x 5 = 40 loads/step forever.
# Budget 42 s/step steady, ~48 s/step during pool warm-up:
#     400 steps x 42 s = 16,800 s = 4h40  + ~8 min boot  = ~4h50
# -> --time=06:00:00 with ~1h of margin.  On a 2h10 slot instead, use
#    MAXSTEPS=150 (110 min stepping / 42 s = 157): 150 is the MINIMUM
#    interpretable length, because gan_disc_start_step=20 +
#    gan_warmup_steps=25 means NOTHING about the GAN is visible before
#    step 45 and the videos would be identical to gantune_w2's.
PIXW=${PIXW:-0.0}
if [ "$PIXW" = "0.0" ] || [ "$PIXW" = "0" ]; then
  W2PIX_MODE=probe
  MAXSTEPS=${MAXSTEPS:-120}
else
  W2PIX_MODE=graded
  MAXSTEPS=${MAXSTEPS:-400}
fi
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
echo "GANTUNE arm=$DARM ckpt=$ODE_CKPT steps=$MAXSTEPS nodes=$SLURM_NNODES $(date)"
# The parent's [GANTUNE-KNOBS] echo was a HARDCODED string that did not
# interpolate and went stale the moment an arm changed a knob.  Both
# echoes below are GREPPED OUT OF THE ACTUAL OVERRIDE BLOCK, so they
# cannot disagree with what is launched.
_OVR_BLOCK() { sed -n '/--override/,/run_name=/p' "$0"; }
OVR_KEYS=$(_OVR_BLOCK | grep -oE '^ +[A-Za-z_][A-Za-z0-9_.]*=' | tr -d ' =')
OVR_DUPS=$(printf '%s\n' "$OVR_KEYS" | sort | uniq -d)
if [ -n "$OVR_DUPS" ]; then
  echo "[GANTUNE-FATAL] DUPLICATE override keys -- last-wins would silently"
  echo "[GANTUNE-FATAL] override the earlier one. Refusing to launch: $OVR_DUPS"
  exit 3
fi
echo "[GANTUNE-KEYCHECK] $(printf '%s\n' "$OVR_KEYS" | wc -l) override keys, 0 duplicates"
echo "[GANTUNE-KNOBS] $(_OVR_BLOCK | grep -oE '^ +(gan_loss_weight|gan_lr|gan_updates_per_step|gan_warmup_steps|gan_disc_start_step|gan_critic_warmup_steps|ladd_r1_gamma|ladd_r1_every_n_steps|ladd_fake_backbone_grad_scale)=[^ ]+' | sed 's/^ *//' | tr '\n' ' ')"
echo "[W2PIX-KNOBS] $(_OVR_BLOCK | grep -oE '^ +(gan_pixel_texture_enabled|surrogate_critic_enabled|pix_finish_grad_enabled|pix_gan_weight|pix_gan_lr|pix_r1_gamma|pix_r1_sigma|pix_crop_lat|pix_crops_per_step|pix_frames_per_crop|pix_lat_frames_per_crop|pix_decode_batch|pix_decode_border_trim|pix_band_count|pix_real_pool_warm_updates|pix_poscontrol_every|texture_tripwire_gt_ref)=[^ ]+' | sed 's/^ *//' | tr '\n' ' ')"
echo "[W2PIX-MODE] mode=$W2PIX_MODE PIXW=$PIXW MAXSTEPS=$MAXSTEPS poscontrol=${PIX_POSCTL:-0}"
# Fail fast on the one thing that would make the arm a silent no-op: the
# flash t=60 x0 is the fake source (pix_finish_grad_enabled=false), and
# trainer::_pix_select_fake_latents RAISES if flash-DMD is off.  The base
# recipe sets flash_dmd_enabled=true; assert it rather than trust it.
_OVR_BLOCK | grep -qE '^ +flash_dmd_enabled=true' || {
  echo "[GANTUNE-FATAL] pix_finish_grad_enabled=false needs flash_dmd_enabled=true"; exit 5; }
echo "[W2PIX-PREFLIGHT] flash_dmd_enabled=true confirmed (fake = flash t=60 x0)"

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
    ladd_gt_vs_fake_enabled=false \
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
    gan_loss_weight=2.0 \
    gan_lr=2e-5 \
    gan_updates_per_step=5 \
    gan_warmup_steps=25 \
    ladd_r1_gamma=10.0 \
    ladd_r1_every_n_steps=1 \
    ladd_fake_backbone_grad_scale=0.2 \
    gan_pixel_texture_enabled=true \
    surrogate_critic_enabled=false \
    pix_finish_grad_enabled=false \
    pix_gan_weight=$PIXW \
    pix_gan_lr=1e-4 \
    pix_r1_gamma=200 \
    pix_r1_sigma=0.01 \
    pix_crop_lat=[24,32] \
    pix_crops_per_step=8 \
    pix_frames_per_crop=3 \
    pix_lat_frames_per_crop=2 \
    pix_decode_batch=2 \
    pix_decode_border_trim=8 \
    pix_band_count=3 \
    pix_reals_per_fake=1 \
    pix_real_pool_windows=4096 \
    pix_real_pool_refresh=8 \
    pix_real_pool_warm_updates=128 \
    pix_loss_form=nsgan \
    pix_poscontrol_every=${PIX_POSCTL:-0} \
    disc_holdout_probe_every=0 \
    texture_tripwire_gt_ref=true \
    $DEXTRA \
    run_name=dmd10k_${DARM}_j${SLURM_JOB_ID}

echo "GANTUNE $DARM completed $(date)"
