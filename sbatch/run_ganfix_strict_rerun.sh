#!/bin/bash
set -euo pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM
: "${HOLDER:?set HOLDER to a running holder job ID}"

export PORTOFF=${PORTOFF:-9985}
export RUNSTAMP=${RUNSTAMP:-$(date +%H%M%S)}
export DARM=ganfix_strict_rerun
export MAXSTEPS=${MAXSTEPS:-200}
export CKPT_EVERY=100000
export STAT_ANCHOR=1.0
export ODE_CKPT=logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt
export AR_HEAD=1.0
export TF_HEAD=0.0
export WAVELET=false

RUN_DIR="logs/dmd10k_${DARM}/dmd10k_${DARM}_h${HOLDER}_${RUNSTAMP}"
export DEXTRA="auto_resume=false boundary_vae_roundtrip=true max_rolls_per_ride=6 rolling_random_depth_enabled=true rolling_random_depth_min=2 rolling_random_depth_max=6 dmd_only_first_chunk_per_ride=false dmd_42f_rolling_sup_new=true streaming_chunk_size=18 num_chunks_roll_forward=3 streaming_min_new_frame=9 streaming_max_length=60 dmd_rolling_ctx_last_rung=true rollout_viz_source=finish dmd_42f_clean_match_enabled=true dmd_42f_clean_match_max_drift_frames=12 dmd_42f_clean_match_min_improve=0.15 dmd_42f_clean_drift_enabled=true dmd_42f_clean_drift_chunks=3 dmd_42f_clean_drift_couple_rope=true dmd_42f_clean_match_drift_compose=true dmd_42f_fix_clean_counterpart=true dmd_supervise_roll_mode=random dmd_sample_at_rungs=false dmd_score_t_min=20 dmd_score_t_max=980 timestep_shift=5.0 fake_score_init_from_teacher=true resume_load_fake_score=false fake_score_ema_weight=0.0 dmd_loss_start_step=50 dmd_loss_warmup_steps=20 dmd_normalization_enabled=true dmd_normalization_denom_floor=1e-6 dmd_ar_normalization_source=ar lr=2e-6 fake_lr=4e-7 streaming_fake_updates_per_gen=4 ema_weight=0.99 ema_start_step=0 carn_seam_affine_lambda=0.5 gan_enabled=true flash_dmd_enabled=true flash_dmd_gan_t=60 gan_loss_weight=0.03 gan_lr=1e-5 gan_updates_per_step=5 gan_disc_start_step=20 gan_critic_warmup_steps=20 gan_warmup_steps=25 gan_warmup_shape=linear ladd_proj_dim=256 ladd_scalar_output=true ladd_freeze_projector_mixing=true ladd_use_csm=true ladd_use_lateral_proj=false ladd_use_prompt_cond=false ladd_cmap_dim=0 ladd_stat_head_enabled=false ladd_wavelet_hf_enabled=false ladd_wavelet_hf_augment=false ladd_gt_transition_enabled=true ladd_gt_transition_match=true ladd_gt_transition_match_k=4 ladd_gt_transition_match_pool=8 ladd_gt_transition_match_max_real=12 ladd_gt_transition_mean_equalize=false ladd_gt_transition_cross_equalize=false ladd_real_pool_cross_ride=4096 ladd_real_pool_push_per_ride=8 ladd_disc_sample_t=true ladd_disc_t_min=20 ladd_disc_t_max=980 ladd_disc_timestep_shift=5.0 ladd_real_match_fake_t=false ladd_r1_gamma=1.0 ladd_r1_every_n_steps=2 ladd_r1_once_per_step=true ladd_r1_num_samples=6 ladd_r1_normalize_tokens=false ladd_r2_gamma=1.0 ladd_r2_every_n_steps=2 ladd_r2_phase_offset=1 ladd_r2_sigma=0.01 ladd_diff_aug_policy=flip ladd_gt_transition_action_blind=false save_full_checkpoint=false eval_checkpoint_path=${RUN_DIR}/eval_step0200.pt real_teacher_causal_mask=false fake_score_causal_mask=false dmd_grad_target_norm=1.0 gan_grad_telemetry_every=25"

bash sbatch/_fgan_holder.sh > logs/exp_${DARM}.err 2>&1

# Chain the canonical fixed-route 60 s eval immediately (probe right after
# training — never batch evals at the end).
ARM="$DARM" CKPT="${RUN_DIR}/eval_step0200.pt" bash sbatch/run_eval60.sh
