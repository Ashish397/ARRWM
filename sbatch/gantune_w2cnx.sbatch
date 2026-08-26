#!/bin/bash
# VARIANT: w2sam  --  gantune_w2 PLUS the SAM2 LATENT-SURROGATE critic
# (WP-SURROGATE / B3).  EXACT copy of sbatch/gantune_w2.sbatch (the
# already-measured w=2.0 tuning point) with ONE mechanism added, so the
# delta vs gantune_w2 is attributable to the surrogate alone.
#
# WHAT THE MECHANISM IS (read the code, do not trust the name):
#   A frozen SAM2 Hiera-B+ PIXEL discriminator (model/r3gan_sam2.py:
#   frozen encoder + 11.22 M trainable ADM 2D heads) is the TEACHER.  It
#   never touches the generator.  A small trainable LATENT critic
#   (model/latent_texture_critic.py, 17.4 M, d_model 512 x 4 blocks) is
#   DISTILLED onto that teacher's value field AND its input gradient
#   field, on VAE latents.  The generator's texture term is then served
#   by the LATENT critic every step -- so the generator gets a
#   pixel-quality signal WITHOUT a VAE decode inside its own backward.
#   The decode is paid only by the distiller, once every
#   pix_teacher_refresh_every steps, under checkpointing.  That is what
#   "latent surrogate" means here: latent SURROGATE FOR a pixel critic.
#
# THE INERT-CONFIG TRAP, AND WHY IT DOES NOT APPLY TO THIS ARM:
#   trainer/causal_action_forcing_train.py:2116 warns that
#   surrogate_critic_enabled=true with gan_pixel_texture_enabled=false
#   builds a critic that never trains and is never consumed.  That
#   warning is GUARDED on surrogate_teacher_backbone=="pixel"
#   (:2096-2098) and is correct ONLY for that backbone.  With
#   backbone=sam2 the trainer runs "surrogate-only mode", admitted at all
#   THREE gates:
#     :17722  G-term call site        -> `or surrogate_critic_enabled`
#     :18094  distillation call site  -> `elif surrogate_critic_enabled`
#     :8290   G-helper top            -> `or _sur_on`
#   and :8045 makes the distiller SELF-SUPPLY its real pool via
#   _pix_pool_fill (the pixel D-loop, the pool's only other producer,
#   does not run).  So gan_pixel_texture_enabled STAYS FALSE here: turning
#   it on would add an entire second, from-scratch pixel PatchGAN
#   adversarial signal and destroy the attributable delta.  Proof of
#   non-inertness is DEMANDED FROM THE RUN, not assumed -- see
#   "PROVE IT" below.
#
# INHERITED w2 HYPOTHESIS (unchanged): gan_loss_weight=2.0 with the R1
# guard on (gamma 10.0, every_n 1) so the disc cannot run away while the
# generator feels the stronger adversarial term.  RISK: if the disc is
# still informative-but-saturating, weight 2.0 turns a bad gradient into a
# dominant one -- watch `cos` (a sustained strongly negative gan/dmd cosine
# = the GAN is fighting DMD) and the 7-chunk videos for texture blow-up.
#
# PROVE IT (the run must show ALL of these, or the arm is a no-op):
#   stdout : "PRETRAINED surrogate teacher built: backbone=sam2"  (:2216)
#   stdout : t_dloss= / t_sep= / c_vloss= / c_corr= on the step line
#            (:3799-3892) -- these keys EXIST ONLY past the no-teacher
#            guard inside _maybe_run_surrogate_distillation, so their
#            presence is positive proof the distillation actually ran.
#   wandb  : gen/train/surrogate_consumed        == 1
#            gen/train/surrogate_distill_ran     == 1
#            gen/train/surrogate_teacher_is_pretrained == 1
#            gen/train/surrogate_g_weighted      != 0   <-- THE one that
#              was exactly 0.0 in the 2026-08-24 smokes, because the
#              `backbone` line had been lost and distillation silently
#              never ran (trainer:7988-7995 documents that incident).
#            gen/train/pix_g_applied == 1 and pix_g_via_gen_gan_loss == 1
#            ABSENT: surrogate_distill_no_teacher, surrogate_distill_no_reals
#
# WEIGHT CHOICE -- pix_gan_weight=0.25.  NOTE the surrogate's applied
# weight is `pix_gan_weight` (resolved in _pix_gen_weight:7936 through
# resolve_gan_weight(strict=True) -- NO DEFAULT, absence RAISES).  There
# is no `surrogate_g_weight` CONFIG key; that name exists only as a LOG
# key (latent_texture_critic.py:1193).  The term is folded into
# gen_gan_loss at :17846 AFTER _compute_r3gan_losses has already applied
# gan_loss_weight, so pix_gan_weight is an ABSOLUTE weight directly
# comparable to the LADD term's 2.0.  Scales: the LADD g-term is a
# logistic loss ~O(0.7) x 2.0 ~= 1.4; the surrogate g-term is a RAW value
# -critic(z).mean() ~O(1) (it tracks the SAM2 teacher's logits, which sit
# at |d_real| ~ 1-3 once the heads train).  0.25 therefore puts the
# surrogate at roughly 15-20 % of the LADD term: loud enough to move
# `ratio` measurably above the already-measured w2 trajectory, far too
# small to take the generator over.  0.01 (the 2026-08-24 smoke value) is
# 200x below LADD and would be indistinguishable from zero -- that is the
# "vanish" failure.  1.0+ would put it at parity with LADD and the arm
# would stop being a w2 delta.  The exact calibration number for the NEXT
# arm is published by this one as gen/train/pix_gan_grad_norm_unweighted
# plus the A7 unweighted-ratio probe.
#   pix_r1_gamma=0.0 : R1 belongs to the from-scratch PIXEL critic, which
#   is OFF here.  0.0 is the explicit, legal "deliberately off" value
#   (_pix_resolve_r1_gamma:6297); absence RAISES and the shipped 1.0 is
#   REFUSED as measured-inert.
#   pix_teacher_refresh_every=4 : the affordability knob.  The SAM2 heads
#   D-update runs EVERY step; the expensive graph-on decode + SAM2
#   fwd/bwd teacher refresh runs every 4.  Both 1 and 4 were exercised
#   2026-08-24.
#
# TEACHER FOOTPRINT -- RE-SIZED AFTER A MEASURED OOM (run 1, 2026-08-25
# 16:49-17:08 on holder 6126936, 95.00 GiB cards).  Run 1 used the
# validated smoke geometry (surrogate_sam2_resolution=512, the YAML's
# pix_crops_per_step=8) and DIED at step 36:
#   torch.OutOfMemoryError in latent_texture_critic.py:698
#   compute_teacher_targets -> torch.autograd.grad, i.e. the graph-on
#   checkpointed VAE decode + SAM2 Hiera fwd/bwd.
# Matched-step evidence that isolates the cost (w2sam vs gantune_w2, same
# holder, same steps):
#   step  0/5/10/15 : 46.49/49.63/49.63/49.86  vs  46.05/49.18/49.19/49.41
#                     -> the surrogate's RESIDENT cost is +0.45 GB (the
#                        frozen SAM2 encoder).  Nothing.
#   step 20         : 91.27                    vs  48.84
#                     -> +42.4 GB TRANSIENT on the first distillation
#                        step.  THAT is the whole problem.
#   step 25         : 91.80  (94.58 GiB in use; the grad-check audit OOM'd
#                        non-fatally here, try/except at trainer:8201)
#   step 36         : fatal -- the 7-chunk 168-frame sample render raises
#                        the floor and the +42 GB transient no longer fits.
# Step time was NOT the problem: 31.9 s/step measured over steps 21->31
# WITH the GAN and the surrogate both live, vs gantune_w2's 33.50.
# The transient is dominated by SAM2 Hiera activations retained for the
# input-gradient pass: n_images = pix_crops_per_step x pix_frames_per_crop
# at surrogate_sam2_resolution^2.  So it is cut on BOTH axes:
#   pix_crops_per_step   8 -> 4   (halves n_images; 4 is the value the
#                                  trainer's own _pix_pool_fill docstring
#                                  uses as its worked A20 example)
#   surrogate_sam2_resolution 512 -> 256  (tokens /4)
# => transient ~/8 on the SAM2 term and ~/2 on the decode term, i.e.
#    ~42 GB -> ~8 GB, giving a projected peak ~73 GB against the 95 GB
#    card and ~65 GB of gantune_w2 baseline peak at full rolling depth.
#   surrogate_grad_check_every 25 -> 1000 (effectively off): the audit
#   costs ONE MORE teacher fwd+grad at exactly the worst moment and was
#   the first thing to OOM (non-fatally) in run 1.  It is telemetry; it
#   does not belong on the critical path of a memory-bound arm.
# COST TO THE SCIENCE, stated plainly: the teacher now sees 256-px crops,
# not 512.  Hiera-B+ is "happiest at 1024" (r3gan_sam2.py:348-351), so
# this is a genuinely weaker teacher than the 2026-08-24 preflight's.
# The alternative was no run at all on a 95 GB card.
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
#SBATCH --job-name=gt-w2cnx
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=03:30:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
DARM=${DARM:-gantune_w2cnx}
ODE_CKPT=${ODE_CKPT:-/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt}
# WALL-CLOCK ARITHMETIC (2-node holder, measured on the CARN arms this is
# forked from -- v6na step21->step71 = 1629 s / 50 = 32.6 s/step; v6seam
# = 32.0 s/step; 6allon = 33.1 s/step, all INCLUDING the sample_interval=15
# video renders).  Stripping CARN can only make it faster, so budget the
# CARN number: 200 steps x 33 s = 6600 s = 110 min, + ~5 min boot
# (srun overlay + model load + first ride) = ~115 min ~ the 2 h target.
#
# RE-SIZED FOR THIS ARM (2026-08-25 16:44).  The holder this lands on has
# 2:11:57 = 132 min of wall left, NOT 6 h.  Budget:
#     132 min wall - 8 min boot/model-load - 7 min margin = 117 min stepping
# gantune_w2 measured 33.50 s/step (steps 21->161, incl. the
# sample_interval=15 video renders).  The surrogate adds, per step, a SAM2
# heads NS-logistic D-update (2 no-grad VAE decodes of 8 crops x 3 frames
# + Hiera-B+ fwd/bwd at 512 res) plus a latent-critic forward, and every
# 4th step the graph-on teacher refresh.  The 2026-08-24 post-fix smokes
# measured that whole package at ~0.6 s/step at N=4 (60 steps: 20m46s
# with distillation vs 20m08s without) -- but those ran gan_enabled=false,
# so budget an order of magnitude more conservatively at ~+10 s/step:
#     assume 45 s/step  ->  117 min = 7020 s  ->  156 steps
# MAXSTEPS=150 therefore finishes CLEANLY with margin at 45 s/step (112
# min) and early at 40 s/step (100 min).  Even at a pessimistic 55 s/step
# it wall-clocks around step 128 -- still far past the step>=45 the
# `ratio`/d_loss medians need.  Finishing cleanly beats hitting 200.
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
echo "GANTUNE arm=$DARM ckpt=$ODE_CKPT steps=$MAXSTEPS nodes=$SLURM_NNODES $(date)"
# ---------------------------------------------------------------------
# [GANTUNE-KNOBS].  gantune_w2.sbatch carried a HARDCODED echo here that
# interpolated NOTHING, so it could silently disagree with the override
# block below -- the config-vs-log mismatch class this campaign keeps
# finding (two more instances fixed 2026-08-25).  Replaced by two things
# that CANNOT drift, because both read THIS FILE at run time:
#   (1) a duplicate-key preflight over the override block.  This config
#       system is LAST-WINS, so a duplicated key is a SILENT override.
#       Fails the job loud rather than running a config nobody can read
#       off the script.
#   (2) an echo whose values are extracted from the override block, not
#       retyped next to it.
# ---------------------------------------------------------------------
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
echo "[SAM2SUR-KNOBS] $(_OVR_BLOCK | grep -oE '^ +(gan_pixel_texture_enabled|surrogate_critic_enabled|surrogate_teacher_backbone|surrogate_sam2_checkpoint_path|surrogate_sam2_config_path|surrogate_sam2_resolution|surrogate_sam2_pad_to_square|surrogate_sam2_preserve_aspect|surrogate_sam2_lr|surrogate_sam2_updates_per_step|pix_gan_weight|pix_r1_gamma|pix_crops_per_step|pix_frames_per_crop|pix_teacher_refresh_every|surrogate_grad_check_every)=[^ ]+' | sed 's/^ *//' | tr '\n' ' ')"
# SAM2 asset preflight: both paths have NO DEFAULT in the trainer
# (:2175 raises without them) and the config path is a SYMLINK into the
# installed sam2 package, which r3gan_sam2.py:132-146 converts back to
# the Hydra-relative name.  A broken symlink here would surface as an
# opaque Hydra error 4 minutes into model load, so check it up front.
for _f in /scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_base_plus.pt \
          /scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_b+.yaml; do
  [ -e "$_f" ] || { echo "[GANTUNE-FATAL] missing SAM2 asset: $_f"; exit 4; }
done
echo "[SAM2SUR-ASSETS] ok (ckpt + config symlink resolve)"

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
    gan_pixel_texture_enabled=false \
    surrogate_critic_enabled=true \
    surrogate_teacher_backbone=convnext \
    surrogate_sam2_checkpoint_path=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_base_plus.pt \
    surrogate_sam2_config_path=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_b+.yaml \
    surrogate_sam2_resolution=256 \
    pix_crops_per_step=4 \
    surrogate_sam2_pad_to_square=true \
    surrogate_sam2_preserve_aspect=true \
    surrogate_sam2_lr=5e-4 \
    surrogate_sam2_updates_per_step=8 \
    pix_gan_weight=2.0 \
    pix_r1_gamma=0.0 \
    pix_teacher_refresh_every=4 \
    surrogate_grad_check_every=1000 \
    $DEXTRA \
    run_name=dmd10k_${DARM}_j${SLURM_JOB_ID}

echo "GANTUNE $DARM completed $(date)"
