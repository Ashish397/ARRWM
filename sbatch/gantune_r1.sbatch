#!/bin/bash
# VARIANT: r1  --  SINGLE AXIS: R1 gradient penalty hardened.
# ladd_r1_gamma 1.0 -> 10.0 and ladd_r1_every_n_steps 2 -> 1.
# HYPOTHESIS: the observed d_loss -> 0.005 with a 4-6 logit gap is an
# UNDER-REGULARISED discriminator.  R1 is the standard anti-saturation term;
# firing it on EVERY D-update at 10x gamma should hold d_loss in 0.25-0.55 and
# the gap at 1-3 for the whole run, at the price of a slower-learning disc.
# SECOND PURPOSE -- INSTRUMENTATION: in all four unmatched 08-25 arms
# (ladd_gt_transition_match=false) `r1` reads EXACTLY 0 at every logged step
# while the matched arms (v5b/v6f/6all*) read 1e-4..0.3.  every_n=1 makes R1
# due on every step, so a still-zero `r1` here is proof the penalty never
# reaches the unmatched pair mode -- i.e. the simple GAN runs unregularised.
# Cost is unchanged per call (ladd_r1_num_samples stays 3) so the memory
# profile matches base.
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
#SBATCH --job-name=gt-r1
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
DARM=${DARM:-gt_r1}
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
echo "GANTUNE arm=$DARM ckpt=$ODE_CKPT steps=$MAXSTEPS nodes=$SLURM_NNODES $(date)"
echo "[GANTUNE-KNOBS] gan_loss_weight=0.3 gan_lr=2e-5 gan_updates_per_step=5 gan_warmup_steps=25 ladd_r1_gamma=10.0 ladd_r1_every_n_steps=1 ladd_fake_backbone_grad_scale=0.2"

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
    gan_loss_weight=0.3 \
    gan_lr=2e-5 \
    gan_updates_per_step=5 \
    gan_warmup_steps=25 \
    ladd_r1_gamma=10.0 \
    ladd_r1_every_n_steps=1 \
    ladd_fake_backbone_grad_scale=0.2 \
    $DEXTRA \
    run_name=dmd10k_${DARM}_j${SLURM_JOB_ID}

echo "GANTUNE $DARM completed $(date)"
