#!/bin/bash
set -euo pipefail

# =====================================================================
# ONE-FORCING (Option D) — SMOKE
#
# First GPU execution of the OF arm. See docs/ONE_FORCING_PORT.md.
#
# WHAT THIS SMOKE IS FOR. The register-token disc head is hosted on the
# TRAINABLE fake_score and its classify forward runs through
# ``_bidir_forward_with_action_tokens`` (action tokens interleaved, so
# 1561 tokens/frame, not 1560). That forward has NEVER executed — flash
# attention has no CPU path, so no test could reach it. This smoke exists
# to answer, in order:
#   1. does the classify forward run at all at real geometry?
#   2. does the D-loss fold into critic_loss without a DDP desync across
#      the 2 nodes (two grad-on fake_score forwards, one backward)?
#   3. is the peak memory survivable?
#   4. is of_logit_gap NON-ZERO and moving? A gap pinned at ~0 is the
#      One-Forcing Fig-4 collapse signature and means the arm is dead.
# Quality is NOT judged here.
#
# LADD is OFF (gan_enabled=false). One GAN at a time — the trainer
# hard-raises if both are on.
#
# boundary_vae_roundtrip_keep_graph=true (added after run r2).
# r2 died on roll 2, all 8 ranks, in the OF G-term's fail-loud guard:
# the rolled chunk had NO autograd graph. Root cause was NOT the GAN —
# ``generate_next_chunk`` rebuilt the chunk with a ``torch.cat`` executed
# inside the ``no_grad`` block that wraps the boundary VAE round-trip, so
# every roll with overlap came back graph-free. That also made the
# streaming DMD generator loss a CONSTANT on every roll after the first
# (the phase-LoRA ghost anchor hid it: generator_loss.requires_grad
# stayed True and gen_backward_skipped read 0.0). The flag keeps the VAE
# forwards under no_grad and moves only the cat out. Default is false =
# byte-identical to every previous run, so this is the ONE line that
# makes this arm's DMD gradient behave the way the recipe intends.
# Watch ``dmd_sup_band_graph_on`` (must be 1.0) and ``of_band_graph_on``.
#
# Heads: TF only. AR-only teacher heads wasted three runs on 08-23.
# Stat anchor: 1.0. GAN_REDESIGN calls it retired, but _roll_holder.sh
# records that removing it collapses the run to black by ~step 61, and
# every recent working arm ran it at 1.0. Keeping it means the GAN is
# measured against the same baseline the other arms used; dropping it
# would confound "OF failed" with "the recipe collapsed".
# =====================================================================

cd /scratch/u6ex/as1748.u6ex/ARRWM
: "${HOLDER:?set HOLDER to a running holder job ID}"

export PORTOFF=${PORTOFF:-7311}
export RUNSTAMP=${RUNSTAMP:-$(date +%H%M%S)}
export DARM=of_smoke
export MAXSTEPS=${MAXSTEPS:-6}
export CKPT_EVERY=100000
export STAT_ANCHOR=1.0
export ODE_CKPT=logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt
export TF_HEAD=1.0
export AR_HEAD=0.0
export NNODE=${NNODE:-2}

RUN_DIR="logs/dmd10k_${DARM}/dmd10k_${DARM}_h${HOLDER}_${RUNSTAMP}"

# Rolling streaming recipe = the current working baseline (_roll_holder /
# ganfix lineage), with the LADD GAN swapped out for One-Forcing.
export DEXTRA="auto_resume=false boundary_vae_roundtrip=true boundary_vae_roundtrip_keep_graph=true max_rolls_per_ride=6 rolling_random_depth_enabled=true rolling_random_depth_min=2 rolling_random_depth_max=6 dmd_only_first_chunk_per_ride=false dmd_42f_rolling_sup_new=true streaming_chunk_size=18 num_chunks_roll_forward=3 streaming_min_new_frame=9 streaming_max_length=60 dmd_rolling_ctx_last_rung=true rollout_viz_source=finish dmd_42f_clean_match_enabled=true dmd_42f_clean_match_max_drift_frames=12 dmd_42f_clean_match_min_improve=0.15 dmd_42f_clean_drift_enabled=true dmd_42f_clean_drift_chunks=3 dmd_42f_clean_drift_couple_rope=true dmd_42f_clean_match_drift_compose=true dmd_42f_fix_clean_counterpart=true dmd_supervise_roll_mode=random dmd_sample_at_rungs=false dmd_score_t_min=20 dmd_score_t_max=980 timestep_shift=5.0 fake_score_init_from_teacher=true resume_load_fake_score=false fake_score_ema_weight=0.0 dmd_loss_start_step=50 dmd_loss_warmup_steps=20 dmd_normalization_enabled=true dmd_normalization_denom_floor=1e-6 lr=2e-6 fake_lr=1e-5 streaming_fake_updates_per_gen=4 ema_weight=0.99 ema_start_step=0 carn_seam_affine_lambda=0.5 real_guidance_scale=0.0 flash_dmd_enabled=true flash_dmd_gan_t=60 gan_enabled=false gan_of_enabled=true gan_of_g_weight=0.03 gan_of_d_weight=0.03 gan_of_feature_layers=[21,29] gan_of_blocks_per_token=1 gan_of_block_ffn_dim=2048 gan_of_block_num_heads=12 gan_of_head_hidden_dim=1536 gan_of_head_num_layers=1 gan_of_head_dropout=0.0 gan_of_t_min=20 gan_of_t_max=980 gan_of_timestep_shift=5.0 gan_of_relativistic=false gan_of_shared_noise=true gan_of_r1_weight=0.0 gan_of_r2_weight=0.0 gan_of_disc_start_step=0 gan_of_warmup_steps=0 gan_of_fake_source=pred_image gan_of_real_source=aligned_gt gan_of_telemetry_every=1 save_full_checkpoint=false real_teacher_causal_mask=false fake_score_causal_mask=false dmd_grad_target_norm=1.0"

# OF_EXTRA: append-only escape hatch so a smoke can exercise the GAN early
# (the default recipe gates DMD until step 50, and the OF G term fires only
# when the DMD scorer fires -- a short smoke otherwise tests nothing).
export DEXTRA="$DEXTRA ${OF_EXTRA:-}"

bash sbatch/_fgan_holder.sh 2>&1 | tee logs/exp_${DARM}_${RUNSTAMP}.err
