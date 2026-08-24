#!/bin/bash
# EXPOSURE SMOKE (WP-PIXGAN, researcher directive 2026-08-24): mechanics run of
# the 10x D-exposure build — pix_crops_per_step=8 (config), pix_gan_updates_per_step
# follows gan_updates_per_step=5 (DEXTRA) => 120 LIVE fakes / 120 band-matched
# fresh reals per step. NO fake ring (withdrawn by researcher: negatives are
# non-stationary by direction; D must detect the CURRENT collapse).
# w=0.01 / gamma=0.0 remain MECHANICS placeholders — no texture claim.
# Readouts: step-time delta vs r3 (12/12), decode counts, exposure totals,
# depth histogram, A20 1:1 at the new shape, A21 support climb.
# smoke_pixgan — MECHANICS smoke of the landed pixel-GAN path (T3-B/C as
# currently in-tree): transition critic OFF per B1 spec, pixel gate ON,
# flash fake, pix_gan_weight=0.01 (mechanics only — NOT the calibrated arm;
# gamma/weight calibration remain researcher-held per the Option-C hold).
# 60 steps, stat anchor OFF. Breaks route to WP-PIXGAN or main-agent fix.
set -euo pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM
: "${HOLDER:?set HOLDER to a running holder job ID}"

export PORTOFF=${PORTOFF:-9990}
export RUNSTAMP=${RUNSTAMP:-$(date +%H%M%S)}
export DARM=smoke_pixexp
export MAXSTEPS=${MAXSTEPS:-60}
export CKPT_EVERY=100000
export STAT_ANCHOR=0.0
export ODE_CKPT=logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt
export AR_HEAD=0.0
export TF_HEAD=1.0
export WAVELET=false

RUN_DIR="logs/dmd10k_${DARM}/dmd10k_${DARM}_h${HOLDER}_${RUNSTAMP}"
export DEXTRA="auto_resume=false boundary_vae_roundtrip=true max_rolls_per_ride=6 rolling_random_depth_enabled=true rolling_random_depth_min=2 rolling_random_depth_max=6 dmd_only_first_chunk_per_ride=false dmd_42f_rolling_sup_new=true streaming_chunk_size=18 num_chunks_roll_forward=3 streaming_min_new_frame=9 streaming_max_length=60 dmd_rolling_ctx_last_rung=true rollout_viz_source=finish dmd_42f_clean_match_enabled=true dmd_42f_clean_match_max_drift_frames=12 dmd_42f_clean_match_min_improve=0.15 dmd_42f_clean_drift_enabled=true dmd_42f_clean_drift_chunks=3 dmd_42f_clean_drift_couple_rope=true dmd_42f_clean_match_drift_compose=true dmd_42f_fix_clean_counterpart=true dmd_supervise_roll_mode=random dmd_sample_at_rungs=false dmd_score_t_min=20 dmd_score_t_max=980 timestep_shift=5.0 fake_score_init_from_teacher=true resume_load_fake_score=false fake_score_ema_weight=0.0 dmd_loss_start_step=0 dmd_loss_warmup_steps=20 dmd_normalization_enabled=true dmd_normalization_denom_floor=1e-6 dmd_ar_normalization_source=ar lr=2e-6 fake_lr=4e-7 streaming_fake_updates_per_gen=4 ema_weight=0.99 ema_start_step=0 carn_seam_affine_lambda=0.5 gan_enabled=false flash_dmd_enabled=true flash_dmd_gan_t=60 gan_loss_weight=0.03 gan_lr=1e-5 gan_updates_per_step=5 gan_disc_start_step=20 gan_critic_warmup_steps=20 gan_warmup_steps=25 gan_warmup_shape=linear ladd_proj_dim=256 ladd_feature_blocks=[0,2,4,8,29] ladd_scalar_output=false ladd_freeze_projector_mixing=true ladd_use_csm=true ladd_use_lateral_proj=false ladd_use_prompt_cond=false ladd_cmap_dim=0 ladd_stat_head_enabled=false ladd_wavelet_hf_enabled=false ladd_wavelet_hf_augment=false ladd_gt_transition_enabled=true ladd_gt_transition_match=true ladd_gt_transition_match_k=8 ladd_gt_transition_match_pool=100000 ladd_gt_transition_match_max_real=16 ladd_gt_transition_mean_equalize=false ladd_gt_transition_cross_equalize=false ladd_real_pool_cross_ride=8192 ladd_real_pool_push_per_ride=32 ladd_disc_micro_batch_groups=4 ladd_disc_sample_t=false ladd_real_match_fake_t=false ladd_r1_gamma=1.0 ladd_r1_every_n_steps=2 ladd_r1_num_samples=6 ladd_r1_normalize_tokens=true ladd_diff_aug_policy=flip,translation ladd_gt_transition_action_blind=false save_full_checkpoint=false eval_checkpoint_path=${RUN_DIR}/eval_step0200.pt real_teacher_causal_mask=false fake_score_causal_mask=false dmd_grad_target_norm=1.0 gan_grad_telemetry_every=25 texture_tripwire_every=25 gan_real_diversity_log=true gan_pixel_texture_enabled=true pix_finish_grad_enabled=false pix_gan_weight=0.01 pix_r1_gamma=0.0 disc_holdout_probe_every=25"
# pix_r1_gamma=0.0: EXPLICIT R1-OFF (WP-PIXGAN, 10:4x). gamma was never
# calibrated on the norm-free critic (yaml ships null -> fail-loud at
# construction, which killed the first 200-step launch). 0.0 is the
# honest mechanics setting; calibrate later from this run's measured
# train/pix_r1_grad_sq_mean (running mean, not a single row).

bash sbatch/_fgan_holder.sh > logs/exp_${DARM}.err 2>&1

