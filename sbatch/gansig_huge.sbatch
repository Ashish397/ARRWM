#!/bin/bash
# VARIANT: gansig_huge  --  the MAXIMUM real distribution this tree can serve,
# with EVERY memory guard still active.  Re-planned 2026-08-25 after the
# coordinator's three-coupling audit (GAN_SIGNAL_WRITEUP.md H5b).
#
# THE DESIGN CHOICE, STATED UP FRONT.  The audit offered two designs, both of
# which run guard-free because ``all_pairs`` disables ``_pos_guard_on``:
#   (a) wide_real + all_pairs + ladd_fake_sample_source=dmd
#   (b) ladd_gt_transition_all_pairs=true only
# This arm is NEITHER, for a reason that is checkable in one grep:
#
#   ``_pos_guard_on`` gates the POSITIONAL D-update ONLY.  The MATCHED
#   D-update is a different code path with its OWN guards, and it honours
#   BOTH ``ladd_disc_micro_batch_groups`` (via
#   _ladd_disc_update_microbatched, ``_micro_groups=_micro_groups``) AND
#   ``ladd_r1_num_samples`` (the A17 fix on the inline matched path,
#   "honour ladd_r1_num_samples on the INLINE matched path too").
#
# So the guard-free hazard is a property of ``all_pairs``, not of "widening
# the real set".  Take the matched path instead and you get a real set that is
# an ORDER OF MAGNITUDE wider than either (a) or (b) WITH the micro-batching
# and R1 subsampling intact.  Note also what (a) and (b) actually buy:
#   (b) does not widen the real set AT ALL -- gt_transition all_pairs leaves
#       the pool at the same 5 fixed positional GT pairs and only multiplies
#       the comparisons 5x5.  It is a reduction change, not a data change.
#   (a) SHRINKS it: under dmd the band is ~3 chunks, so n_pairs 6 -> 3.
# Neither answers "make the set huge".  This does.
#
# WHAT MOVES vs sbatch/gantune_w1.sbatch (the running weight-ladder base):
# EXACTLY ONE THING -- where the discriminator's real transitions come from.
# ONE pair mode (gt_transition), as in w1.  ladd_fake_backbone_grad_scale
# stays at w1's 0.2 (1 mode x 5 updates = the neutral value).  No gt_vs_fake,
# no all_pairs anywhere, no wide_real, no dmd band.
#
#   ladd_gt_transition_match=true  -> per-fake CONTENT-MATCHED retrieval from
#       the whole ride window instead of the fixed positional counterpart.
#   ladd_gt_transition_match_pool=100000 -> clamps to n_cand, so the top-M
#       relevant set IS the whole window (M = 22 transitions), and it is
#       RESAMPLED FRESH ON EVERY ONE of the 5 D-updates (the _match_select
#       salt), not once per step.
#   ladd_real_pool_cross_ride=256 + ladd_real_pool_push_per_ride=8
#       -> the CROSS-RIDE replay ring.  Half of each fake's K real slots
#       (Kh = K//2) are drawn UNIFORMLY from a 256-entry per-rank ring of GT
#       transition slabs accumulated across ~32 PREVIOUS RIDES, each carrying
#       its own actions and prompt embeds.  THIS IS THE ONLY MECHANISM IN THE
#       TREE BY WHICH THE DISC'S REAL SET EVER LEAVES THE CURRENT RIDE.
#   ladd_gt_transition_match_k=12 / _max_real=12 -> 12 reals per fake:
#       6 ride-local matched + 6 cross-ride.  _cap_m = 12//2 = 6 == _Ksel, so
#       the 6 matched slots map to 6 DISTINCT rows with no cap collapse.
#
# REAL-SET ARITHMETIC (per rank, per D-update; B=1, npb=3,
# streaming_chunk_size=18 -> 6 chunks -> 5 gt_transition pairs; S=5):
#   Kk = min(12, n_cand=22) = 12    Kh = 6    _Ksel = 6
#   _cap_m = max(1, 12//2) = 6   -> n_mu <= 6      (ride-local matched rows)
#   _budget = 12 - 6 = 6         -> _R  = min(len(ring), max(6, 6)) = 6
#   n_uniq = 6 + 6 = 12 distinct reals per D-update, RESAMPLED x5 per step.
#
#   gantune_w1 (positional gt_transition) : 5 rows, the SAME 5 on all 5
#       D-updates, pool = 5 FIXED positions, 1 ride  ->   5 distinct/step
#   gansig_huge                           : 12 rows, fresh x5
#                                                       -> up to 60/step
#       pool = 22 ride-local + 256 cross-ride = 278 candidates, ~33 rides.
#   Across 8 ranks: 40 -> up to 480 distinct real rows/step.
#   RpGAN comparisons/step: 25 (5 pairs x 5 updates) -> 300 (5 fakes x 12
#       block-diagonal reals x 5 updates).
#
# MEMORY -- ALL GUARDS ON, and this is the point of the design.
#   ladd_disc_micro_batch_groups=8       HONOURED (matched micro-batcher)
#   ladd_gen_guidance_micro_batch_groups=8  HONOURED
#   ladd_r1_num_samples=3                HONOURED (A17 inline-matched fix)
#   ladd_lazy_cand_disc=true             candidate list not pinned across the
#       D-loop + gen-side + deferred backward (6all_gside_diagnosis.md)
#   ladd_defer_disc_update=true          inherited
# The combined disc forward is 2*n_uniq + n_fake rows: 12 unique reals vs
# smk_v6f's 16 (which peaked 89.8 GB WITH the full CARN stack).  This arm
# carries NO CARN and NOT the register readout.
# EXPECTED PEAK: 68-76 GB on 96 GB cards, anchored on TWO measurements today:
#   gantune_w1   max step_peak = 64.89 GB (holdersmoke_gantune_w1_h6126937)
#   gansig_real  max step_peak = 69.76 GB (holdersmoke_gansig_real_h6120979)
#     -> gansig_real's +6 real and +6 fake single-chunk rows (36 frame-rows)
#        cost +4.9 GB, i.e. ~0.14 GB per frame-row.  This arm adds +7 real
#        rows of 6 frames = 42 frame-rows over w1 => ~+6 GB => ~71 GB.
# WATCH step_peak FROM STEP 20 (first D-update) ONWARD and report the
# TRAJECTORY, not a final number.  An OOM here is data.
#
# R1 HELD AT w1's RATE -- ladd_r1_unified_cadence=false -> TRUE.  NOT optional:
# on the legacy path the MATCHED branch fires ONE global-latch R1 per due step
# while the POSITIONAL branch fires M*S (the FIRE-RATE TABLE above
# _ladd_r1_block_due).  Left false, this arm would run at 1 R1/step against
# w1's 5 -- silently gutting the anti-saturation regulariser under test.
# unified=true restores M*S = 1x5 = 5/step, bit-for-bit w1's budget.
#
# NO SILENT NO-OPS.  ladd_gt_vs_fake_n_real / ladd_gt_transition_n_real are
# DEAD CODE in this tree (read off args, consumed only by a mutual-exclusion
# raise; no branch ever samples N real chunks).  They are NOT set here.
# ladd_gt_transition_all_pairs is NOT set: it is mutually exclusive with
# match (the matched branch RAISES if both are on) and it would disable
# _pos_guard_on.
#
# ACTION-CONDITIONING NOTE.  The gt_vs_fake+wide_real asymmetric-action
# shortcut (WRITEUP b3/H5b) CANNOT arise here: gt_vs_fake is off entirely.
# The matched gt_transition reals take their own co-located pool_act_full
# slice while the fake keeps the act_lo0 origin -- the same arrangement that
# already ships in all seven matched production arms, so this arm introduces
# no new class of risk relative to production.
#
# ============ WHY match_pool=100000 HERE IS NOT THE v6f BUG ============
# The coordinator's audit is right that v6f's match_k=8 / match_pool=100000
# made the ride-local draw a RANDOM one: _M = max(Kk, min(_M, n_cand)) clamps
# 100000 to n_cand=22, and sampling Kk of 22 uniformly is a wide draw, not a
# nearest-neighbour retrieval.  v6f WANTED retrieval, so for v6f that was a
# bug, and smk_v6fmit's match_k=1 / match_pool=3 is the right fix.
# THIS ARM WANTS THE OPPOSITE.  Its brief is "make the real set huge": a wide,
# decorrelated real distribution is the mechanism under test (the saturation
# diagnosis is that 5 perfectly-correlated GT rows are memorisable).  So the
# wide draw is the TREATMENT here, deliberately, and it is kept honest by
# running on the MATCHED code path purely because that path is the only one
# that (a) resamples fresh per D-update, (b) can reach the cross-ride ring,
# and (c) keeps micro-batching + R1 subsampling (unlike all_pairs).
# CONSEQUENCE FOR READING THE TELEMETRY: ladd_match_dist_ratio will read
# ~1.0 on THIS arm BY DESIGN.  On smk_v6fmit / the production arms ~1.0 is a
# FAILURE (matching no better than random); here it is the expected value and
# must NOT be reported as a defect.  The number that matters here is
# gan_real_unique_rides, not dist_ratio.
# =======================================================================
#
# CODE CONTRACT: this arm carries the five 2026-08-25 fixes that are going
# onto all ten production arms (srcmd5 5c8faae419ec9e294cb23733daa34c6e), so
# it is judged under the SAME contract as everything else:
#   ladd_gside_checkpoint_recover=true   (FIX 1, ~30 GiB back on the G side --
#       this INVALIDATES the 68-76 GB estimate above DOWNWARD; expect lower)
#   ladd_action_origin_track_slice=true  (FIX 2 -- act_lo0 = the latent slice
#       origin, so ladd_act_lag -> 0.  This REMOVES the real/fake action
#       asymmetry noted below, i.e. the confound is fixed, not merely flagged)
#   + ladd_action_origin_log, gan_health_show_r1_fire_rate,
#     gan_health_show_hold_step, ladd_match_distance_log,
#     ladd_backbone_grad_scale_check
# BASELINE CAVEAT: gantune_w1 (64.89 GB) and gansig_real (69.76 GB) were
# measured WITHOUT these fixes.  Memory and act-lag comparisons against them
# are cross-contract; the saturation comparison (d_loss / gap) is still valid.
# DO NOT RUN THIS BEFORE smk_v6fmit PASSES -- that smoke is what gates these
# seven never-GPU-tested flags.
#
# WHAT TO READ (beyond the GAN-HEALTH judging table below):
#   train/ladd_ring_size  -> must CLIMB toward 256.  0 = ring never armed.
#   train/ladd_ring_n     -> unique ring reals in the last draw; expect 6.
#   train/ladd_match_n_cand_gtxn -> 22.  train/ladd_match_k_gtxn -> 12.
#   train/ladd_match_n_real_gtxn -> ~12 (distinct rows actually forwarded).
#   train/gan_real_unique_rides  -> THE HEADLINE.  >1 proves the real set
#         left the current ride; ~1 means the ring is inert and this failed.
#   train/gan_real_repeat_rate, _ring_slot_frac, _unique_windows.
#   [LADD-MATCH-FALLBACK] must NOT appear (it means the pool was missing and
#         the step silently used positional pairing).
#   [LADD-POS-GUARD] must NOT appear (it would mean a guard was dropped).
#   train/ladd_disc_deferred_updates should read 1 (ONE pair mode).
#   train/ladd_act_lag -> 0 (FIX 2 took).  r1_rate in [GAN-HEALTH].
#   NO logging.error from the backbone grad-scale checker: this arm is
#   single-mode at 0.2, which is exactly 1/(1 x 5).
#   Only the _gtxn suffix is expected; there is no _gt mode in this arm.
# =====================================================================
# GAN-SIGNAL SWEEP (2026-08-25).  analysis/gan_tuning/GAN_SIGNAL_WRITEUP.md
# Forked from sbatch/gantune_w1.sbatch; gan_loss_weight=1.0 and
# ladd_r1_gamma=10.0 are HELD AT w1's VALUES so these arms are directly
# comparable to the running weight ladder.  The ONLY things that move are the
# pair-mode / One-Forcing knobs listed in this arm's paragraph above.
#
# FACT ESTABLISHED FROM CODE FIRST (do not re-litigate):  the disc ALREADY
# sees real dataset latents today.  gt_transition's real side is
# ``gt_latents_window.detach()`` (trainer:10069) = ``state["ride_latents_window"]
# [:, chunk_lo:chunk_hi]`` (trainer:17180) = ``ride["latents"]``, the VAE-encoded
# dataset footage (trainer:18302/18324).  Real and fake are re-noised
# SYMMETRICALLY at disc_t_int = flash_dmd_gan_t = 60 (trainer:11666-11670).
# So these arms are NOT "give the GAN real data" -- they are "give it MORE and
# DIFFERENTLY-SAMPLED real data, at a second granularity".
# ---------------------------------------------------------------------------
# !! TWO HAZARDS MEASURED/TRACED AFTER THIS SCRIPT WAS WRITTEN -- READ BEFORE RUN
#
# (1) all_pairs DISABLES THE MEMORY GUARDS on the gt_vs_fake D-update.
#     trainer/causal_action_forcing_train.py:13872-13898 -- ``_pos_guard_on``
#     requires ``not _all_pairs_mode``, so with all_pairs ON both
#     ladd_disc_micro_batch_groups=8 and ladd_r1_num_samples=3 are IGNORED for
#     that mode and it falls back to the SINGLE-BATCH FULL-R1 path.  The
#     trainer prints [LADD-POS-GUARD] once when this happens -- GREP FOR IT.
#     Measured baseline: gansig_real (same two modes, guards ACTIVE) peaked at
#     69.76 GB on a 96 GB card.  This arm removes the guards for one of those
#     two modes, so the peak WILL be higher and OOM is a live possibility.
#     WATCH ``step_peak`` in the [ROLL] lines from step 20 on.
#     NOTE ``ladd_gt_vs_fake_wide_real`` CANNOT be used without all_pairs
#     (trainer:11389-11394 makes _all_pairs_mode a hard precondition), so this
#     hazard is not avoidable while keeping the wide real set.
#
# (2) wide_real corrects the ACTION ORIGIN ON THE REAL SIDE ONLY.
#     trainer:11918-11922 -- ``lo_r`` becomes the exact absolute frame under
#     wide_real, but ``lo_f`` keeps the lagged ``act_lo0 = cf_state`` origin
#     unconditionally.  On rolls 2/3/4 (lag 9/18/27 frames) the REAL rows are
#     action/latent-consistent and the FAKE rows are not, which hands the disc
#     an action-consistency shortcut unrelated to visual realism.  A disc that
#     finds it posts a great logit gap and teaches the generator nothing.
#     => DO NOT read this arm's gap / d_loss against gansig_real at face value.
#        Judge it on the SAMPLE VIDEOS.
#     => ladd_fake_sample_source=dmd removes the asymmetry (it re-bases
#        act_lo0 for BOTH sides) -- that is gansig_of, run that first.
#     => gt_transition is IMMUNE: its real_acts and fake_acts are byte-identical
#        tensors by construction (trainer:12226-12233).  Only gt_vs_fake+wide_real
#        can create the shortcut.
# ---------------------------------------------------------------------------
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
#SBATCH --job-name=gs-huge
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
DARM=${DARM:-gansig_huge}
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
echo "GANSIG-HUGE arm=$DARM ckpt=$ODE_CKPT steps=$MAXSTEPS nodes=$SLURM_NNODES $(date)"
echo "[GANSIG-KNOBS] arm=$DARM gan_loss_weight=1.0 gan_lr=2e-5 gan_updates_per_step=5 gan_warmup_steps=25 ladd_r1_gamma=10.0 ladd_r1_every_n_steps=1 ladd_r1_unified_cadence=TRUE ladd_r1_num_samples=3 ladd_fake_backbone_grad_scale=0.2 ladd_gt_transition_enabled=true ladd_gt_transition_match=TRUE ladd_gt_transition_match_k=12 ladd_gt_transition_match_pool=100000 ladd_gt_transition_match_max_real=12 ladd_lazy_cand_disc=TRUE ladd_real_pool_cross_ride=256 ladd_real_pool_push_per_ride=8 ladd_gt_transition_all_pairs=false ladd_gt_vs_fake_enabled=FALSE ladd_gt_vs_fake_all_pairs=false ladd_gt_vs_fake_wide_real=false ladd_adjacent_chunks_enabled=false ladd_fake_sample_source=flash ladd_readout=ladd(default) ladd_disc_micro_batch_groups=8 ladd_gen_guidance_micro_batch_groups=8 gan_real_diversity_log=true ladd_gside_checkpoint_recover=TRUE ladd_action_origin_track_slice=TRUE ladd_action_origin_log=true gan_health_show_r1_fire_rate=true gan_health_show_hold_step=true ladd_match_distance_log=true ladd_backbone_grad_scale_check=true srcmd5_expected=5c8faae419ec9e294cb23733daa34c6e"

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
    ladd_r1_num_samples=2 \
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
    ladd_r1_gamma=10.0 \
    ladd_r1_every_n_steps=1 \
    ladd_fake_backbone_grad_scale=0.2 \
    ladd_gt_vs_fake_weight=1.0 \
    ladd_gt_transition_weight=1.0 \
    ladd_gt_vs_fake_enabled=false \
    ladd_gt_vs_fake_all_pairs=false \
    ladd_gt_vs_fake_wide_real=false \
    ladd_gt_transition_all_pairs=false \
    ladd_gside_checkpoint_recover=true \
    ladd_action_origin_track_slice=true \
    ladd_action_origin_log=true \
    gan_health_show_r1_fire_rate=true \
    gan_health_show_hold_step=true \
    ladd_match_distance_log=true \
    ladd_backbone_grad_scale_check=true \
    ladd_gt_transition_match=true \
    ladd_gt_transition_match_k=12 \
    ladd_gt_transition_match_pool=100000 \
    ladd_gt_transition_match_max_real=6 \
    ladd_lazy_cand_disc=true \
    ladd_real_pool_cross_ride=256 \
    ladd_real_pool_push_per_ride=8 \
    $DEXTRA \
    run_name=dmd10k_${DARM}_j${SLURM_JOB_ID}

echo "GANSIG $DARM completed $(date)"
