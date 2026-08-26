#!/bin/bash
# VARIANT: w2style  --  gantune_w2 PLUS the LPIPS-VGG PERCEPTUAL TERM.
# EXACT copy of sbatch/gantune_w2.sbatch (the already-measured w=2.0
# tuning point) with ONE mechanism added, so the delta is attributable.
#
# =====================================================================
# READ THIS FIRST -- WHAT "STYLE" DOES AND DOES NOT EXIST IN THIS TREE
# =====================================================================
# The objective is MATCHING THE FRODOBOTS DASHCAM APPEARANCE
# DISTRIBUTION -- its grain, colour response, contrast, lens character
# and texture statistics -- NOT producing more high-frequency energy.
# A mechanism that adds sharpness while moving AWAY from the dataset's
# look is a FAILURE.  Audited against that bar, the tree contains:
#
#   DOES NOT EXIST (would need new code, cannot be turned on by config):
#     * Gram-matrix / AdaIN STYLE LOSS.  The only Gram/AdaIN code is
#       OFFLINE EVAL: grids/eval/style_shift.py (VGGStyle,
#       gram_distance), analysis/style_shift/axis2_local_style.py,
#       utils/style_shift_detect.py.  ZERO imports from trainer/,
#       model/ or pipeline/.  (model/anti_collapse.py's "gram" is a CxC
#       matrix for stable-rank eigenvalues, not a style loss.)
#     * The HF-fraction / anisotropy / kurtosis MATCHING loss that
#       TEXTURE_GAN_DESIGN §8 calls "arm J / B1".  analysis/texture_stats.py
#       has every statistic it would need (hf_power, hv_anisotropy,
#       angular_entropy, hf_kurtosis, haar_HL_LH_ratio) but they are ONLY
#       ever consumed under torch.no_grad() -- by the tripwire and by the
#       §8.1 offline calibration.  Not differentiable-path-connected.
#     * Colour / histogram matching, DISTS, pixel-space Laplacian/Sobel/
#       edge/TV losses.  None exist.
#
#   DOES EXIST, live, flag-gated, reaching the GENERATOR loss:
#     1. LPIPS-VGG perceptual  (lpips_loss_weight)  <-- THIS ARM.
#        Never used: lpips_loss_weight has ZERO hits across all of
#        sbatch/, configs/ and docs/.  It has been sitting at 0.0 since
#        v13.
#     2. STAT ANCHOR (stat_anchor_* / mode=gt_window) -- six per-frame
#        LATENT summary statistics (STD/M2/TV/SOS/M1/MEAN) matched to GT
#        with an MSE-plus-relative-tolerance floor.  ALREADY ON at
#        weight 1.0 in EVERY arm of this sweep (TV_short=0.1,
#        MEAN_short=0.9) and listed as HELD FIXED -- do not vary it here
#        or the sweep loses its control.  It is also far too low-order
#        (six scalars per frame, in latents) to encode "style".
#     3. ANTI-COLLAPSE std_corridor / latent_contrast_drift -- a genuine
#        contrast-MATCHING corridor, but again latent-space and
#        low-order.  Not the mechanism for appearance.
#     4. The PRETRAINED-TEACHER SURROGATE route
#        (surrogate_teacher_backbone = sam2 | dinov2 | convnext).  This
#        IS a learned style critic in disguise -- a frozen encoder
#        trained on real photographs judging real-vs-generated
#        appearance.  It is ALREADY BUILT AND QUEUED as
#        gantune_w2sam / w2dino / w2cnx, so it is NOT rebuilt here.
#
# CONCLUSION, stated plainly: there is NO Gram/AdaIN style-matching loss
# in this codebase, and one cannot be produced config-only.  LPIPS is
# the closest genuine, unexploited, in-tree, pretrained-feature,
# dataset-referenced training-time pressure -- and it is a real
# mechanism, not a stand-in.  What a true style loss would additionally
# need is specified at the end of this header.
#
# =====================================================================
# WHY LPIPS IS THE RIGHT CANDIDATE UNDER THE "DO NOT OPTIMISE FOR
# SHARPNESS" SCREEN
# =====================================================================
#  1. IT CANNOT BE SATISFIED BY BLOCKY OR PIXELATED OUTPUT.  The screen
#     is: "could this be satisfied by blocky, aliased or pixelated
#     output instead of real dataset-like texture?"  For LPIPS the
#     answer is structurally NO.  It is a distance in a FROZEN,
#     PRETRAINED VGG16 feature space -- a learned prior for what real
#     photographs look like -- measured against the ACTUAL decoded GT
#     frame.  Aliased 8-px block structure is FAR from the GT frame in
#     that feature space, so pixelating the output RAISES the loss.
#     Contrast this with the from-scratch pixel PatchGAN (gantune_w2pix),
#     whose local critic has no photographic prior and for which
#     high-contrast hard-edged structure is a cheap way to win.
#  2. IT IS TWO-SIDED BY CONSTRUCTION.  It is a distance minimised AT
#     the dataset frame, not a monotone "more texture is better" term.
#     Overshooting the dataset's high-frequency content increases it
#     exactly as undershooting does.  That is the loss SHAPE the target
#     ("match the distribution") actually calls for.
#  3. IT IS THE REFERENCE COMPANION TERM.  taming-transformers/VQGAN and
#     the SD VAE train with L1 + LPIPS + a pixel PatchGAN.  The Wan
#     latent space we operate in was produced by that recipe.  This
#     campaign has been running the PatchGAN half (in latents, where it
#     is texture-blind) and has NEVER run the LPIPS half.
#
# HONEST LIMITATIONS -- do not oversell this arm:
#   * LPIPS is a PAIRED, SPATIALLY-ALIGNED perceptual distance, not an
#     UNPAIRED style / distribution match.  It asks "is this the same
#     frame", of which "does it look like the dataset" is only a
#     component.
#   * It therefore does NOT directly ask the question that
#     analysis/style_shift/REPORT.md's Axis-2 identified as the real
#     failure: LOCAL TEXTURE REVERSION IN NEWLY GENERATED CONTENT,
#     scored against carried-over REAL content in the SAME frame (the
#     minwmdmd stamped-leaf case, onset 2.31 s, which frame-level CLIP
#     barely moved on).  That is a Gram-shaped, region-masked question.
#   * OVER-FIRE MODE (this is its "failure mode if it over-fires"): at
#     too high a weight it becomes a REGRESSION term fighting DMD's
#     distribution matching.  Once the student's rollout has drifted off
#     the GT trajectory, a frame-aligned perceptual distance pulls
#     toward the conditional mean -> BLUR, ghosting and double edges.
#     That is the OPPOSITE failure direction from the PatchGAN's, which
#     is exactly why the two are worth running as a pair.
#
# STANDING-RULE CONFLICT -- NEEDS RESEARCHER SIGN-OFF BEFORE LAUNCH.
#   docs/GAN_REDESIGN.md §C records: "Texture supervision is ADVERSARIAL
#   ONLY (researcher rule, 2026-08-23).  The anisotropic statistics are
#   instruments, never loss terms."  LPIPS is not one of those
#   statistics, but it IS non-adversarial texture supervision, so this
#   arm sits against the spirit of that rule.  It is built, not
#   launched.  DO NOT SUBMIT WITHOUT AN EXPLICIT OK.
#
# =====================================================================
# CODE FACTS THAT GOVERN THIS ARM (verified, not assumed)
# =====================================================================
#  * Parse: trainer/causal_action_forcing_train.py:3062-3084
#    (lpips_loss_weight default 0.0, lpips_n_frames 2, lpips_crop_size 0,
#    pixel_recon_loss_weight 0.0, pixel_recon_loss_type "l1").
#  * Loss:  :5999 _compute_pixel_perceptual_losses; LPIPS(net="vgg")
#    lazy-built at :5937.  Added to the generator at :18027-18031
#    (`generator_loss = generator_loss + perc_loss`).  Zero weight =
#    ZERO COMPUTE (the graph-on decode is skipped at the call site), so
#    there is no weight-free probe available on this path -- unlike
#    pix_gan_weight, which publishes an UNWEIGHTED ratio at 0.0.  The
#    cheapest probe is therefore a SMALL NON-ZERO weight, which is the
#    default below.
#  * HARD DEPENDENCY, and a real trap: the call site passes
#    `gt_latents_window=gt_window`, and `gt_window` is bound ONLY inside
#    the `if gan_active:` branch at :17731.  `gan_active` is
#    CONFIG-driven (gan_enabled AND r3gan_disc AND r3gan_optimizer, at
#    :17353), not step-driven, so it is True on every step of THIS
#    recipe.  But running this arm with gan_enabled=false would raise
#    UnboundLocalError at step 0.  DO NOT turn the LADD GAN off here.
#  * The `lpips` pip package is REQUIRED (:5934 raises without it).
#    Verified present in the arrwm env; the preflight below re-checks.
#  * lpips_crop_size crops AFTER the decode, so it bounds the VGG cost
#    only, NOT the decode cost.
#
# =====================================================================
# MEMORY -- expected peak ~73 GB on the 95.00 GiB cards
# =====================================================================
#   Baseline gantune_w2 peaks ~65 GB at GAN-ACTIVE steps at full rolling
#   depth (the 46-49 GB step-0..20 readings are EARLY-STEP, not peak).
#   ADDED:
#     * graph-on FULL-FRAME decode of lpips_n_frames LATENT frames.
#       Unlike the pixel arm this is NOT a crop: 2 latent frames at
#       60x104 -> ~8 pixel frames at 480x832 = ~3.2 Mpx.  That is the
#       same order as gantune_w2pix's 8 crops x 8 frames x 192x256 =
#       ~3.1 Mpx, whose decode term was derived at ~7.2 GB from the
#       matched-step w2sam measurement (D + S = 42.4, D/2 + S/8 = 8.0 =>
#       D ~= 7.2, S ~= 35.2).  _vae_decode_grad is torch.utils.checkpoint
#       -wrapped, so ~7 GB is an upper bound.
#     * no_grad GT decode of the same shape -- transient, freed before
#       the VGG pass.
#     * LPIPS-VGG16 fwd/bwd.  At the full 480x832 this would be several
#       GB (first conv alone is 8 x 64 x 480 x 832 x 4 B = 818 MB with
#       grad, over five tapped layers).  lpips_crop_size=256 cuts it to
#       256^2/(480x832) = 16.4% -> ~1 GB.  THIS IS WHY THE CROP IS SET.
#   => 65 + ~7 + ~1 ~= 73 GB, ~22 GB headroom.  w2sam run 1 died with
#   0.4 GB of headroom when the sample_interval=15 render raised the
#   floor at step 36.
#   DOWNSCALE RULE: lpips_n_frames is the ONLY linear axis (it sizes the
#   grad decode).  If step_peak exceeds 82 GB before the first sample
#   render, this arm has nothing to cut except lpips_n_frames=1.
#
# =====================================================================
# READOUT -- and this is a DATASET-DISTANCE readout, not a sharpness one
# =====================================================================
#   texture_tripwire_every=1 is INHERITED from gantune_w2; this arm adds
#   texture_tripwire_gt_ref=true, which publishes train/tripwire_gt_* and
#   train/tripwire_ratio_* -- the same anisotropic battery computed on
#   the frame-matched GT and reported as a RATIO.  The ratio is the
#   two-sided quantity; the raw tripwire_* alone is not.
#     SUCCESS  : tripwire_ratio_hf_power          -> 1.0 from either side
#                tripwire_ratio_hv_anisotropy     -> 1.0
#                tripwire_ratio_angular_entropy   -> 1.0
#                (read JOINTLY -- any one alone is gameable;
#                 analysis/texture_stats.py:texture_battery says so in
#                 its own docstring)
#     BLUR ABORT (this arm's characteristic over-fire):
#                tripwire_ratio_hf_power < 0.70 and still falling, or
#                lpips_loss_raw falling while the 7-chunk videos get
#                visibly softer.  Cut LPW by 3x.
#     BLOCK ABORT (inherited discipline, unlikely here):
#                tripwire_ratio_angular_entropy < 0.90, and post-hoc
#                analysis/vertical_banding/phase_fold.py A8y/A16y above
#                the clean-GT reference (A8y 1.00 / A16y 1.57 in
#                phase_fold_results.json).
#   PLUS: analysis/style_shift/ is the offline appearance instrument for
#   this arm's eval videos (Axis-1 global flip, Axis-1 quality channel,
#   Axis-2 local texture reversion).  Another agent owns extending it --
#   do not duplicate; just feed it the checkpoints.
#
# WEIGHT -- LPW, defaults to a deliberately CONSERVATIVE 0.1.
#   There is no zero-weight probe on this path (zero weight = no
#   compute, no log key), so the first launch doubles as the
#   calibration: read train/lpips_loss_raw and train/lpips_loss_weighted
#   against the DMD loss on the SAME row and target the same 5-20% band
#   the GAN weight targets.  Then bracket:  LPW=0.1 / 0.3 / 1.0.
#   pixel_recon_loss_weight is pinned to 0.0 EXPLICITLY, not left to the
#   default: a pixel L1/L2 is a mean-seeking regression and is the
#   canonical blur generator.  Its exclusion is a decision, and pinning
#   it makes the decision visible in the trace.
#
# WHAT A TRUE STYLE LOSS WOULD NEED (new code -- NOT in this file):
#   Gram/AdaIN matching on a frozen encoder's features between GENERATED
#   crops and REAL DATASET crops, UNPAIRED (the reals are not the fake's
#   own GT counterpart), band-matched on vertical position per A24, with
#   the second-order statistic taken per-region rather than per-frame so
#   it targets analysis/style_shift Axis-2's "new content reverts to a
#   procedural prior" failure.  The encoder and the crop/band/real-pool
#   machinery ALREADY EXIST (grids/eval/style_shift.py::VGGStyle for the
#   Gram; trainer::_pix_pool_fill / _pix_draw_reals / band_plan for the
#   A20/A21-compliant unpaired real supply; _pix_decode_crops_grad for
#   the graph-on crop decode).  The missing piece is a differentiable
#   gram_distance term wired into generator_loss behind a new flag --
#   perhaps 60 lines.  It is the single highest-value CODE item arising
#   from this review.
#
# PROVE IT (the run must show these, or the arm is a no-op):
#   stdout : the LPIPS build line from _get_lpips_model (:5944-5949)
#   wandb  : gen/train/lpips_loss_raw       present and NON-ZERO
#            gen/train/lpips_loss_weighted  present and NON-ZERO
#            gen/train/lpips_n_frames       == 2
#            gen/train/tripwire_ratio_hf_power present
#            ABSENT: gen/train/pixel_recon_loss_raw  (weight pinned 0.0)
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
#SBATCH --job-name=gt-w2style
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
DARM=${DARM:-gantune_w2style}
ODE_CKPT=${ODE_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt}
# WALL-CLOCK ARITHMETIC (2-node holder, measured on the CARN arms this is
# forked from -- v6na step21->step71 = 1629 s / 50 = 32.6 s/step; v6seam
# = 32.0 s/step; 6allon = 33.1 s/step, all INCLUDING the sample_interval=15
# video renders).  Stripping CARN can only make it faster, so budget the
# CARN number: 200 steps x 33 s = 6600 s = 110 min, + ~5 min boot
# (srun overlay + model load + first ride) = ~115 min ~ the 2 h target.
#
# W2STYLE WALL-CLOCK.  gantune_w2 measured 33.50 s/step (steps 21->161,
# incl. the sample_interval=15 renders).  LPIPS adds ONE graph-on
# full-frame decode of 2 latent frames, one no_grad GT decode of the
# same, and a VGG16 fwd/bwd on ~8 crops of 256x256 -- budget +5 s/step:
#     250 steps x 39 s = 9,750 s = 2h43  + ~8 min boot  = ~2h51
# -> --time=04:00:00 with >1h of margin.
# Unlike the GAN arms, LPIPS is ACTIVE FROM STEP 0 (no disc start-step,
# no warm-up ramp), so its own effect is readable early.  250 is chosen
# so the GAN schedule it shares the recipe with is ALSO past its
# gan_disc_start_step=20 + gan_warmup_steps=25 by a wide margin and the
# 7-chunk videos are comparable with the other arms of the sweep.
LPW=${LPW:-0.1}
MAXSTEPS=${MAXSTEPS:-250}
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
# echoes below are GREPPED OUT OF THE ACTUAL OVERRIDE BLOCK.
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
echo "[W2STYLE-KNOBS] $(_OVR_BLOCK | grep -oE '^ +(lpips_loss_weight|lpips_n_frames|lpips_crop_size|pixel_recon_loss_weight|pixel_recon_loss_type|texture_tripwire_every|texture_tripwire_gt_ref|texture_tripwire_frames|gan_pixel_texture_enabled|surrogate_critic_enabled)=[^ ]+' | sed 's/^ *//' | tr '\n' ' ')"
echo "[W2STYLE-MODE] LPW=$LPW MAXSTEPS=$MAXSTEPS"
# Preflight 1: the `lpips` pip package has NO fallback -- _get_lpips_model
# raises 40+ minutes into a run otherwise (the SAM2-asset lesson).
python -c "import lpips" 2>/dev/null || {
  echo "[GANTUNE-FATAL] the 'lpips' package is not importable in this env"; exit 6; }
echo "[W2STYLE-PREFLIGHT] lpips importable"
# Preflight 2: the LPIPS call site reads gt_window, which the trainer
# binds ONLY inside `if gan_active:` -- gan_enabled=false would
# UnboundLocalError at step 0.
_OVR_BLOCK | grep -qE '^ +gan_enabled=true' || {
  echo "[GANTUNE-FATAL] lpips_loss_weight>0 requires gan_enabled=true (gt_window binding)"; exit 7; }
echo "[W2STYLE-PREFLIGHT] gan_enabled=true confirmed (gt_window is bound)"

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
    lpips_loss_weight=$LPW \
    lpips_n_frames=2 \
    lpips_crop_size=256 \
    pixel_recon_loss_weight=0.0 \
    pixel_recon_loss_type=l1 \
    texture_tripwire_gt_ref=true \
    texture_tripwire_frames=2 \
    gan_pixel_texture_enabled=false \
    surrogate_critic_enabled=false \
    $DEXTRA \
    run_name=dmd10k_${DARM}_j${SLURM_JOB_ID}

echo "GANTUNE $DARM completed $(date)"
