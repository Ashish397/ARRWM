#!/bin/bash
#SBATCH --job-name=of-200-8n
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# ONE-FORCING (Option D) — 200-step 8-node arm, off the ODE-KL student.
#
# The faithful port of One-Forcing (arXiv 2605.23458). See
# docs/ONE_FORCING_PORT.md for the full spec and the five divergences
# from our old LADD arm that this fixes.
#
# THE ONE-LINE THESIS: the discriminator is hosted on the TRAINABLE
# fake_score critic, not the frozen real_score teacher. The disc backbone
# is co-trained every step by the denoising loss on the current student's
# samples, so its feature basis tracks the student instead of being a
# frozen teacher basis that is blind to exactly DMD's own mistakes.
# Obeys GAN_REDESIGN_TWO's binding rule (never build the disc from
# real_score) by a different route than options A/B/C.
#
# HOW TO JUDGE THIS RUN, in order:
#   1. of_logit_gap. One-Forcing's Fig. 4 is the whole ablation: a healthy
#      disc holds |d_real - d_fake| ~ 1.5 and MOVING; a collapsed one sits
#      at ~0 from step 1. If this pins at 0, the arm is dead and no image
#      metric matters.
#   2. of_gan_dmd_ratio_unweighted — the GAN's share of the generator
#      gradient. The old LADD arms read 0.05-1.2%, i.e. the GAN was
#      irrelevant. We want a real contribution here.
#   3. gen/dmd_mae_gate_m_fake (student MAE, lower better) — the
#      standing quality instrument. NOT the raw d_real-d_fake gap.
#   4. Only then: the videos.
#
# Recipe = the current rolling streaming baseline (_roll_holder / ganfix
# lineage) with the LADD GAN swapped out for One-Forcing. LADD is OFF;
# the trainer hard-raises if both GANs are on.
# Heads: TF only (AR-only heads wasted three runs on 08-23).
# Stat anchor 1.0 — deliberate deviation from "retired", justified in
# docs/ONE_FORCING_PORT.md: at 0 the recipe collapses to black by ~61 and
# we could not separate "GAN did nothing" from "recipe collapsed".
# =====================================================================

set -euo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export ARRWM_ACTION_ENCODER=pca_raw
export TMPDIR=/tmp
unset LOCALDIR APPTAINER_CACHEDIR
# Pre-built manifest — skips the ~50-min per-launch ride scan.
export ARRWM_MANIFEST_PICKLE=/scratch/u6ex/as1748.u6ex/ARRWM/analysis/.dmd_weunz_manifest_min69.pt

RUNSTAMP=${RUNSTAMP:-$(date +%m%d_%H%M%S)}
DARM=of200
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/${DARM}/${DARM}_j${SLURM_JOB_ID}_${RUNSTAMP}
WANDB_SAVE_DIR=wandb
CONFIG=configs/action_forcing_phase3_dmd.yaml
ODE_CKPT=logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt

mkdir -p "$LOGDIR" logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn NCCL_DEBUG=WARN NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

echo "ONE-FORCING 200-step 8-node arm; ckpt=$ODE_CKPT; $(date)"
echo "LOGDIR=$LOGDIR"
echo "Judge by: of_logit_gap (must be non-zero and moving), then of_gan_dmd_ratio_unweighted, then dmd_mae_gate_m_fake."
echo "Nodes: $SLURM_NNODES  Total GPUs: $(( SLURM_NNODES * 4 ))"

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
    run_name=${DARM}_j${SLURM_JOB_ID} \
    max_steps=200 \
    checkpoint_interval=100 \
    keep_last_n_checkpoints=99 \
    save_full_checkpoint=true \
    auto_resume=false \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_weunz \
    holdout_zarr_list=configs/holdout_rides.txt \
    holdout_eval_root=/projects/u6ex/fbots/frodobots_encoded_weu_holdout \
    holdout_eval_mode=cycle \
    sample_7chunk_enabled=false \
    max_ride_frames=900 \
    max_ride_frames_random=false \
    seed=1234 \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    action_dims=[0,1] \
    action_critic_dims=[0,1] \
    denoising_step_list=[1000,625,357.142857,208.333333] \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=21 \
    local_attn_size_schedule=[[0,21]] \
    num_workers=0 \
    sample_fps=16 \
    boundary_vae_roundtrip=true \
    infinity_rope=true \
    seed_prefill_mode=real \
    kv_cache_seed_headroom=true \
    real_teacher_causal_mask=false \
    fake_score_causal_mask=false \
    real_guidance_scale=0.0 \
    dmd_context_clean_frames=9 \
    dmd_loss_weight=1.0 \
    dmd_loss_start_step=50 \
    dmd_loss_warmup_steps=20 \
    dmd_normalization_enabled=true \
    dmd_normalization_denom_floor=1e-6 \
    dmd_grad_target_norm=1.0 \
    dmd_mae_gate_enabled=false \
    dmd_sample_at_rungs=false \
    dmd_score_t_min=20 \
    dmd_score_t_max=980 \
    timestep_shift=5.0 \
    stat_anchor_loss_weight=1.0 \
    stat_anchor_mode=gt_window \
    stat_anchor_match_k=2 \
    stat_anchor_M2_short_weight=0.1 \
    stat_anchor_TV_short_weight=0.1 \
    stat_anchor_M1_short_weight=0.0 \
    stat_anchor_M2_long_weight=0.0 \
    stat_anchor_TV_long_weight=0.0 \
    stat_anchor_SOS_short_weight=0.0 \
    stat_anchor_STD_short_weight=0.0 \
    stat_anchor_rel_tol_short=0.0 \
    dmd_asymmetric_scoring_enabled=false \
    dmd_42f_enabled=true \
    dmd_42f_2chunk=false \
    dmd_42f_seed_last=false \
    dmd_42f_gt_after_chunks=1 \
    dmd_42f_num_chunks=4 \
    dmd_42f_gt_anchor=true \
    dmd_42f_rand_sup_slot=false \
    dmd_42f_allsup=false \
    dmd_42f_rolling_sup_new=true \
    dmd_42f_clean_match_enabled=true \
    dmd_42f_clean_match_max_drift_frames=12 \
    dmd_42f_clean_match_min_improve=0.15 \
    dmd_42f_clean_match_forward=false \
    dmd_42f_clean_drift_enabled=true \
    dmd_42f_clean_drift_chunks=3 \
    dmd_42f_clean_drift_couple_rope=true \
    dmd_42f_clean_match_drift_compose=true \
    dmd_42f_fix_clean_counterpart=true \
    dmd_only_first_chunk_per_ride=false \
    dmd_supervise_roll_mode=random \
    dmd_rolling_ctx_last_rung=true \
    max_rolls_per_ride=6 \
    rolling_random_depth_enabled=true \
    rolling_random_depth_min=2 \
    rolling_random_depth_max=6 \
    streaming_chunk_size=18 \
    num_chunks_roll_forward=3 \
    streaming_min_new_frame=9 \
    streaming_max_length=60 \
    streaming_fake_updates_per_gen=4 \
    rollout_viz_source=finish \
    lr=2e-6 \
    fake_lr=4e-7 \
    ema_weight=0.99 \
    ema_start_step=0 \
    fake_score_init_from_teacher=true \
    resume_load_fake_score=false \
    fake_score_ema_weight=0.0 \
    fake_score_lora_enabled=false \
    carn_seam_affine_lambda=0.5 \
    motion_start_threshold=5.0 \
    flash_dmd_enabled=true \
    flash_dmd_gan_t=60 \
    dmd_tf_head_weight=1.0 \
    dmd_ar_head_weight=0.0 \
    real_teacher_train_online=false \
    dmd_frozen_teacher_pass_enabled=false \
    real_score_ema_weight=0.0 \
    aux_teacher_loss_weight=0.0 \
    aux_real_clean_x_source=gt \
    aux_noisy_from_raw_gt=true \
    forward_noiser_enabled=false \
    carn_recurse=false \
    use_8bit_adam=true \
    gen_gradient_checkpointing=true \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    strict_ode_load=true \
    ode_generator_checkpoint=$ODE_CKPT \
    v14_teacher_checkpoint=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    student_phase_lora_enabled=false \
    gan_enabled=false \
    ladd_adjacent_chunks_enabled=false \
    gan_of_enabled=true \
    gan_of_g_weight=0.03 \
    gan_of_d_weight=0.03 \
    gan_of_feature_layers=[21,29] \
    gan_of_blocks_per_token=1 \
    gan_of_block_ffn_dim=2048 \
    gan_of_block_num_heads=12 \
    gan_of_head_hidden_dim=1536 \
    gan_of_head_num_layers=1 \
    gan_of_head_dropout=0.0 \
    gan_of_t_min=20 \
    gan_of_t_max=980 \
    gan_of_timestep_shift=5.0 \
    gan_of_relativistic=false \
    gan_of_shared_noise=true \
    gan_of_r1_weight=0.0 \
    gan_of_r2_weight=0.0 \
    gan_of_disc_start_step=0 \
    gan_of_warmup_steps=0 \
    gan_of_fake_source=pred_image \
    gan_of_real_source=aligned_gt \
    gan_of_telemetry_every=25

echo "ONE-FORCING 200-step arm finished on $(date)"
