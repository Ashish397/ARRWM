#!/bin/bash
#SBATCH --job-name=arrwm_p4_E3_1_SAv3_FT_v3
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# FT_v3 = FT_v2 rolling rematch with a STABILISED DMD target.
#   * Online teacher FROZEN (not disabled): real_teacher_train_online
#     STAYS true so the rank-32 UNLOCKED LoRA from j3d is still built,
#     loaded from the checkpoint, and applied as the DMD real score —
#     but aux_teacher_loss_weight=0 + aux_teacher_separate_backward=false
#     short-circuit the aux training pass entirely (no LoRA grad, no
#     1.3B aux forward/backward), and real_score_ema_weight=0 +
#     real_teacher_lr=0 keep it bit-frozen. NB: real_teacher_train_online
#     =FALSE would instead DISCARD the rank-32 LoRA and revert the real
#     score to plain merged-v14 (losing the entire j-series unlock) —
#     that is the trap we are avoiding.
#   * Freeing the aux pass (~10-15 GB) lets the GAN go back to full
#     strength: ladd_pairs_per_step=7 (v6e), up from FT_v2's OOM-cut 3.
#   * fake_score_ema_weight=0.95 KEPT (the j1/j2/j3 EMA setting the
#     d-fleet showed best).
#   * Rollout machinery UNDER REVIEW — unchanged here pending the
#     rollout reconsideration; do not submit until that is settled.
# =====================================================================
# train_arrwm_p4_E3_1_nsallsup_SAv3_FT_v1 = PHASE-2/4 ROLLING TRAINING
# on the v8h3 base (naive forward CARN-former). Resumes from h3's
# step-200 checkpoint (pre-staged below) and rolls each ride INDEFINITELY
# instead of re-seeding every step:
#   * Per ride: seed 6 GT chunks (cf=18) -> iter 1 rolls the full 7-chunk
#     window -> every later step rolls 3 NEW chunks (9f) on the
#     persistent KV cache (self-forcing) and trains the latest 7-chunk
#     window [4 ctx | 3 new].
#   * num_chunks_roll_forward=4 + streaming_min_new_frame=12 =
#     deterministic 4-chunk advance (window = [3 kept | 4 new]).
#   * dmd_42f_rolling_sup_new=true: 42f window re-anchors on the NEW
#     frames on rolls k>=2 -> [9f GT ctx | 9f new (supervised) | 3f GT
#     future scaffold]; every rolled chunk DMD-supervised exactly once.
#   * dmd_only_first_chunk_per_ride=false: DMD fires every roll.
#   * MAE-collapse gate: streaming_mae_collapse_threshold (currently 0.0
#     = OFF; set from rolling telemetry) resets the ride when window MAE
#     vs GT crosses it. Rides otherwise reset at end-of-ride.
#   * h3's FN (forward CARN-former, [FN-GT-FORMER]) stays active. The FN
#     teacher_feat training only finds rollout1/rollout2 pairs on each
#     ride's FIRST window (rollout2 covers the setup window); later
#     rolls hit the DDP-safe zero-anchor, so the FN trains at reduced
#     duty during rolling.
#   * 1000 steps total (resume @200 -> +800 rolling), ckpt every 100.
#
# ---- h3 base notes below ----
# ---- legacy e-lineage notes below (mostly describe the FORWARD CARN) ----
# Phase-3 DMD one-step — v8e6 = v8e5 + gan_loss_weight=2.5 + 3h wall to
# reach step 200 in one shot (from scratch). Checkpoints now ALSO save the
# GAN disc + ForwardNoiser, so this run is cleanly resumable.
#
# ---- v8e5 base ----
# Phase-3 DMD one-step — v8e5 = v8e4 but FLATTEN (across all four) instead
# of per-pair.
#
# Same as v8e4 (per-channel mean + std cross-eq) EXCEPT
# ladd_gt_transition_xeq_preserve_delta=FALSE: instead of scaling each
# pair by one shared factor (which preserves the within-pair former->latter
# brightness/contrast transition), every one of the FOUR members -- real
# former, real latter, fake former, fake latter -- is normalized
# INDEPENDENTLY to the common per-channel mean a* and std s*. So the disc
# is FULLY moment-blind: no absolute level, no within-pair transition --
# pure texture only. This is the "across all four" variant vs v8e4's
# "per pair". per-channel + std stay ON.
# Config = v8e4 with preserve_delta=false.
#
# ---- v8e4/v8e3 lineage ----
# v8e4 = v8e3 + STD equalization; v8e3 = v8e2 + per-channel + preserve-delta.
#
# v8e2 confirmed cross-equalization kills the white-collapse (brightness
# flat past the old step-61 point). v8e2 flattened ALL FOUR pair members
# to one level -> the disc was fully brightness-blind (within-pair delta
# gone too). v8e3 flips on ladd_gt_transition_xeq_preserve_delta=true:
# each PAIR is scaled by ONE shared factor to the common level a*, so the
# real/fake ABSOLUTE level is still equalized (no collapse) but the
# within-pair former->latter brightness TRANSITION ratio is PRESERVED ->
# the disc can see, and push the student to match, GT's transition
# brightness behaviour. (To become the hardcoded default.)
# ALSO: ladd_gt_transition_xeq_per_channel=true -> the equalization
# magnitude is computed PER CHANNEL (reduce [F,H,W], keep C) instead of
# one scalar over all channels, so per-channel brightness/colour is fully
# non-discriminative. Fix code is shared; config = v8e2 + these two flags.
#
# ---- v8e2 lineage: CARN BRIGHTNESS-COLLAPSE FIX ----
#
# v8e collapsed to white by ~step 61. Root cause (transition GAN carn
# path): the FN trains on un-normalized rollout1->rollout2 (rollout2 runs
# HOT, +1 drift step) so FN(GT_former) drifts BRIGHTER; the GAN's
# mean_equalize was within-pool only (real and fake equalized separately),
# so the hot real pair sat above the fake pair -> disc learns
# "brighter=real" -> student brightens -> rollout2 brighter -> FN brighter
# -> runaway white. TWO fixes (both in shared trainer code; config is
# otherwise identical to v8e):
#   1) CROSS-EQUALIZED mean_equalize (now the hardcoded default): all four
#      members of each pair row -- real former+latter, fake former+latter
#      -- are scaled to ONE common per-row mean-magnitude. The disc can no
#      longer see ANY magnitude difference (within-pair OR real-vs-fake),
#      so the GAN affects only texture, never brightness. Target is
#      detached -> no magnitude gradient to the student.
#   2) MOMENT-PRESERVING carn (_carn_former): restore the former's
#      per-channel DC + magnitude after the FN, so carn changes only
#      texture STRUCTURE (no stats) and its texture isn't rescaled away by
#      the cross-eq. Belt-and-braces with (1).
#
# Inherits v8b2 (normalize-fix max(|f|,1) + MAE gate + GT-anchor +
# local_attn=24, R1-only GAN). Changes vs v8b2:
#   * Learned ForwardNoiser (CARN) trained to map rollout1 -> rollout2 =
#     the "+1 cartoon shift" (the per-step texture/detail drift of the
#     causal AR rollout). LATENT SPACE — NO wavelet decomposition.
#     forward_noiser_loss_mode=teacher_feat, but the match is now a
#     SLICED-WASSERSTEIN per-token distribution match in the frozen
#     teacher's feature space (NOT mean+std). v8d's mean+std match was
#     inert (fn_tf_loss ~1e-6) because GT<->student moments barely differ
#     and are pinned by the stat anchor; the full SW marginal captures the
#     texture/cartoon shift those moments missed. Step-unconditioned
#     (one generic +1 shift). Separate FN backward, FN optimizer only ->
#     fully DECOUPLED from fake_score.
#   * fake_alt_rollout2_num_seed_chunks=5 = (cf/npb)-1 = 6-1, so rollout2
#     is exactly ONE seed chunk shorter than rollout1 -> a clean +1 drift
#     step (matches the prebuild's max_length_r2 = max_length + npb
#     geometry). rollout2 auto-builds because loss_mode=teacher_feat.
#   * ladd_gt_transition_carn_former=true (1 step): apply the FN ONCE to
#     the FIRST GT chunk of each transition pair -> real anchor =
#     [cartoon-shifted GT former -> clean GT latter] = a SELF-CORRECTING
#     (de-cartoon) transition the student is pushed to match.
#   * GAN at v6e strength: gan_updates_per_step=5, ladd_pairs_per_step=7
#     (RESTORE GAN STRENGTH per the user's instruction). NOTE: rollout2's
#     second no-grad rollout is a real memory tax (it forced the 8d cuts);
#     5/7 + rollout2 may re-OOM. If it does, the lever is GAN strength
#     (5->3) or ladd_pairs_per_step, NOT the CARN.
#   * DMD-strength compensation for the gate kept: dmd_mae_gate_exponent=
#     0.75 (mild concave ramp -> DMD held above the linear weight, fading
#     toward parity, but gentler than the 0.5 sqrt ramp).
#
# 21->24 frame window, 200-step max. GAN ON (ladd_teacher_feat), R1-only.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase3_dmd.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_dmd_one_step
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

mkdir -p "$LOGDIR"
mkdir -p logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export ARRWM_ROPE_DEBUG=1

echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "Starting Phase-3 DMD ONE-STEP (v8e = v8b2 + learned latent-space SW CARN, v6e GAN) on $(date)"
echo "  * allsup: 3 supervised chunks; cf=18 seed + GT-anchor = 7 GT; local_attn=24."
echo "  * 1 roll per ride (fresh seed each step)."
echo "  * GAN: ON (ladd_teacher_feat, matched gt_transition), R1-only, v6e strength (5/7)."
echo "  * CARN-former: learned FN, +1 cartoon shift (latent-space sliced-Wasserstein teacher-feat), rollout2 ON."
echo "  * dmd_normalization: ON, denom floor=1.0 -> divide by max(|f|,1)."
echo "  * MAE gate: ON (r_full=2.0, ema=0.0 [per-sample, no smoothing], min_weight=0.0, exponent=0.75)."
echo "  * Teacher: single frozen merged-v14 (no online, no aux pass)."
echo "  * wandb_project=longlive-phase-3."

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"


# ---- Pre-stage h3's step-200 checkpoint into THIS run's dir so
# auto_resume starts rolling from the trained stationary h3 baseline.
# Skipped when the run dir already holds a checkpoint (requeue case).
# FATAL if no h3 checkpoint exists yet.
RUN_NAME=arrwm_p4_E3_1_nsallsup_SAv3_FT_v3_j${SLURM_JOB_ID}
RUN_DIR=$LOGDIR/$RUN_NAME
mkdir -p "$RUN_DIR"
if ! ls "$RUN_DIR"/phase1_step*.pt >/dev/null 2>&1; then
  # Prefer the LATEST checkpoint from a PREVIOUS rolling run (continue
  # the phase-2 run across resubmits); fall back to the h3 stationary
  # baseline only when no rolling checkpoint exists yet.
  PREV_CKPT=$(ls -t $LOGDIR/arrwm_p4_E3_1_nsallsup_SAv3_FT_v3_j*/phase1_step*.pt 2>/dev/null | head -1 || true)
  if [ -n "$PREV_CKPT" ]; then
    cp "$PREV_CKPT" "$RUN_DIR/"
    echo "Pre-staged PREVIOUS rolling checkpoint: $PREV_CKPT -> $RUN_DIR/"
  else
  # Prefer the CLEAN-DRIFT-unlocked j3e teacher (the real FT_v3 baseline);
  # fall back to the older randx-unlocked j3d (used to smoke the FT_v3
  # machinery before j3e finishes). Either is a valid phase-3 student+
  # teacher checkpoint for resuming.
  H3_CKPT=$(ls -t $LOGDIR/*v8j3e_j*/phase1_step0000200.pt 2>/dev/null | head -1 || true)
  if [ -z "$H3_CKPT" ]; then
    H3_CKPT=$(ls -t $LOGDIR/*v8j3e_j*/phase1_step*.pt 2>/dev/null | head -1 || true)
  fi
  if [ -z "$H3_CKPT" ]; then
    H3_CKPT=$(ls -t $LOGDIR/*v8j3d_j*/phase1_step0000200.pt 2>/dev/null | head -1 || true)
  fi
  if [ -z "$H3_CKPT" ]; then
    H3_CKPT=$(ls -t $LOGDIR/*v8j3d_j*/phase1_step*.pt 2>/dev/null | head -1 || true)
  fi
  if [ -z "$H3_CKPT" ]; then
    echo "FATAL: no v8j3e or v8j3d checkpoint found under $LOGDIR — FT_v3 needs the j3e (clean-drift unlocked) baseline, or j3d for a machinery smoke. Aborting."
    exit 1
  fi
  cp "$H3_CKPT" "$RUN_DIR/"
  echo "Pre-staged unlocked-teacher baseline checkpoint: $H3_CKPT -> $RUN_DIR/"
  fi
fi

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
    max_steps=1000 \
    checkpoint_interval=100 \
    ema_weight=0.0 \
    ema_start_step=1000000 \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_weu \
    holdout_zarr_list=configs/holdout_rides.txt \
    holdout_eval_root=/projects/u6ex/fbots/frodobots_encoded_weu_holdout \
    holdout_eval_mode=cycle \
    max_ride_frames=900 \
    max_ride_frames_random=false \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    max_rolls_per_ride=100000 \
    streaming_max_length=300 \
    gen_gradient_checkpointing=true \
    use_8bit_adam=false \
    num_chunks_roll_forward=2 \
    streaming_min_new_frame=6 \
    dmd_42f_rolling_sup_new=true \
    dmd_42f_clean_drift_enabled=true \
    dmd_42f_clean_drift_start_step=0 \
    dmd_42f_clean_drift_full_step=0 \
    streaming_mae_collapse_threshold=0.0 \
    streaming_mae_collapse_min_chunks=2 \
    rolling_rolls_schedule_every=50 \
    rolling_rolls_schedule_base=2 \
    rolling_rolls_schedule_start_step=200 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=24 \
    num_workers=0 \
    sample_fps=16 \
    boundary_vae_roundtrip=false \
    local_attn_size_schedule=[[0,24]] \
    real_teacher_causal_mask=false \
    real_guidance_scale=0.0 \
    dmd_context_clean_frames=18 \
    dmd_loss_weight=1.0 \
    dmd_loss_start_step=0 \
    dmd_loss_warmup_steps=20 \
    dmd_normalization_enabled=true \
    dmd_mae_gate_enabled=true \
    dmd_mae_gate_r_full=2.0 \
    dmd_mae_gate_ema=0.0 \
    dmd_mae_gate_min_weight=0.0 \
    dmd_mae_gate_exponent=0.75 \
    stat_anchor_loss_weight=1.0 \
    stat_anchor_STD_short_weight=0.0 \
    stat_anchor_STD_long_weight=0.0 \
    stat_anchor_mode=gt_window \
    stat_anchor_match_k=2 \
    stat_anchor_M2_short_weight=0.1 \
    stat_anchor_M2_long_weight=0.0 \
    stat_anchor_SOS_short_weight=0.0 \
    stat_anchor_SOS_long_weight=0.0 \
    stat_anchor_M1_short_weight=3e-9 \
    stat_anchor_M1_long_weight=0 \
    stat_anchor_TV_short_weight=0.1 \
    stat_anchor_TV_long_weight=0.0 \
    stat_anchor_rel_tol_short=0.0 \
    stat_anchor_rel_tol_long=0.0 \
    dmd_asymmetric_scoring_enabled=false \
    dmd_42f_enabled=true \
    dmd_42f_2chunk=true \
    dmd_42f_seed_last=false \
    dmd_42f_gt_after_chunks=3 \
    dmd_42f_rand_sup_slot=false \
    dmd_42f_allsup=true \
    dmd_42f_gt_anchor=true \
    dmd_only_first_chunk_per_ride=false \
    dmd_42f_fix_clean_counterpart=true \
    flash_dmd_enabled=true \
    fake_score_ema_weight=0.95 \
    motion_start_threshold=5.0 \
    fake_score_lora_enabled=false \
    gan_enabled=true \
    gan_backbone=ladd_teacher_feat \
    gan_loss_weight=1.5 \
    gan_lr=2e-5 \
    gan_critic_warmup_steps=20 \
    gan_disc_start_step=20 \
    gan_warmup_steps=25 \
    gan_warmup_shape=linear \
    gan_updates_per_step=5 \
    gan_r1_gamma=2.0 \
    gan_max_grad_norm=10.0 \
    ladd_proj_dim=256 \
    ladd_use_csm=true \
    ladd_use_lateral_proj=false \
    ladd_use_prompt_cond=false \
    ladd_cmap_dim=64 \
    ladd_disc_loss_weight=1.0 \
    ladd_r1_gamma=0.02 \
    ladd_r1_every_n_steps=1 \
    ladd_diff_aug_policy=flip,cutout,translation \
    ladd_pairs_per_step=4 \
    ladd_wavelet_hf_enabled=true \
    ladd_wavelet_hf_drop_ll=true \
    ladd_wavelet_hf_ll_weight=0.0 \
    ladd_wavelet_hf_adapter_init_gain=0.1 \
    ladd_gt_vs_fake_enabled=false \
    ladd_gt_vs_fake_weight=0.0 \
    ladd_gt_transition_enabled=true \
    ladd_gt_transition_weight=1.0 \
    ladd_gt_transition_all_pairs=false \
    ladd_gt_transition_mean_equalize=true \
    ladd_gt_transition_xeq_preserve_delta=true \
    ladd_gt_transition_xeq_per_channel=true \
    ladd_gt_transition_n_real=0 \
    ladd_gt_transition_match=true \
    ladd_gt_transition_match_k=4 \
    real_teacher_train_online=true \
    dmd_frozen_teacher_pass_enabled=false \
    real_score_ema_weight=0.0 \
    real_teacher_lr=0.0 \
    real_teacher_warmup_steps=20 \
    aux_teacher_lora_rank=32 \
    aux_teacher_lora_alpha=64 \
    aux_teacher_loss_weight=0.0 \
    aux_teacher_start_step=0 \
    aux_teacher_loss_warmup_steps=20 \
    aux_teacher_separate_backward=false \
    aux_real_clean_x_source=gt \
    aux_noisy_from_raw_gt=true \
    aux_clean_x_random_window=true \
    aux_clean_x_drift_start_step=40 \
    aux_clean_x_drift_full_step=160 \
    forward_noiser_enabled=true \
    forward_noiser_loss_mode=teacher_feat \
    forward_noiser_step_unconditioned=false \
    forward_noiser_feat_t=0 \
    forward_noiser_sw_n_proj=64 \
    forward_noiser_sw_max_tokens=4096 \
    fake_alt_rollout2_num_seed_chunks=5 \
    forward_noiser_apply_gt_former=true \
    carn_recurse=false \
    forward_noiser_apply_gt_level=1 \
    fn_frontier_pairs=false \
    ladd_adjacent_chunks_enabled=false \
    ladd_adjacent_chunks_weight=0.0 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=${RUN_NAME}

echo "Training completed on $(date)"
