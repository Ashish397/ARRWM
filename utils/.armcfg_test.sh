#!/bin/bash
#SBATCH --job-name=dmd10k
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
# DMD-10K STATIONARY (2026-08-16): reconnect phase-3 DMD to the 10k ODE
# students. Template = train_c8_stat_wave_dmd3_fwd_freal_ode1000.sbatch (the
# validated DMD<-ODE handoff precedent); deltas per
# .claude/dmd_gan_stage_reference.md §7:
#   * 14e lineage: v14e teacher LoRA + pca_raw actions ([0,1]) — the export
#     below is LOAD-BEARING (silent z2/z7 corruption without it).
#   * student rung ladder [1000,625,357.142857,208.333333] (rollkl 20-step
#     grid), NOT the phase-1 default.
#   * DMD ONLY: GAN off (backbone override kept: the constructor check fires
#     even with the GAN off), flash-DMD off, forward-noiser/CARN off,
#     stat anchor 0, teacher FROZEN (plain frozen v14e; no unlock).
#   * dmd_42f_clean_match OFF (in-flight dmd3 variant, deferred).
#   * dmd_context_clean_frames=18 (validated; also keeps the window cache
#     from rolling -> the RoPE-implementation divergence never engages in
#     this stationary stage).
#   * strict_ode_load=true (default false swallows key drift).
#   * NOTE deferred: 9x int-rounded timesteps (<=0.04% shift, immaterial);
#     RoPE decision + commit/prefill A/B belong to the ROLLING stage.
# Submit:
#   sbatch -J dmd10k-<tag> --export=ALL,DARM=<tag>,ODE_CKPT=<path>[,DEXTRA=...]
#     [-N 2 + MAXSTEPS=3 for smokes]  sbatch/train_dmd10k_stat.sbatch
# Judge by gen/dmd_mae_gate_m_fake (falling = student approaching teacher),
# real_score_mae_vs_gt, and teacher_match probes on saved ckpts.
# =====================================================================
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
: "${DARM:?submit with --export=ALL,DARM=<tag>,ODE_CKPT=<path>}"
: "${ODE_CKPT:?submit with --export=ALL,DARM=<tag>,ODE_CKPT=<path>}"
MAXSTEPS=${MAXSTEPS:-200}
DEXTRA=${DEXTRA:-}
# Space-free teacher-head switches. `sbatch --export=...,DEXTRA="a=1 b=2"`
# puts a SPACE inside one exported value, which is exactly the kind of thing
# that mangles silently and trains the wrong config. AR_HEAD / TF_HEAD carry
# a single token each and are assembled here instead.
#   AR_HEAD=1.0            -> dual head (TF + AR)
#   AR_HEAD=1.0 TF_HEAD=0  -> AR-only
[ -n "${AR_HEAD:-}" ] && DEXTRA="$DEXTRA dmd_ar_head_weight=${AR_HEAD}"
[ -n "${TF_HEAD:-}" ] && DEXTRA="$DEXTRA dmd_tf_head_weight=${TF_HEAD}"
echo "DMD10K head config: AR_HEAD=${AR_HEAD:-unset} TF_HEAD=${TF_HEAD:-unset} -> DEXTRA=[$DEXTRA]"

CONFIG=configs/action_forcing_phase3_dmd.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/dmd10k_${DARM}
WANDB_SAVE_DIR=wandb
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export ARRWM_ACTION_ENCODER=pca_raw
# Scan-free start (2026-08-16): pre-built weunz manifest (v4, 2,549 rides
# >=69f, filtered from the .rec_scratch 21.9GB manifest) — skips the ~50-min
# per-launch ride scan entirely.
export ARRWM_MANIFEST_PICKLE=/scratch/u6ex/as1748.u6ex/ARRWM/analysis/.dmd_weunz_manifest_min69.pt
mkdir -p "$LOGDIR" logs
MASTER_ADDR=$(scontrol show hostnames "$(squeue -j 6027557 -h -o %N)" | head -n1)
MASTER_PORT=24997
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn NCCL_DEBUG=WARN NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
echo "DMD10K arm=$DARM ckpt=$ODE_CKPT steps=$MAXSTEPS nodes=$SLURM_NNODES $(date)"

srun --overlap --jobid=6027557 --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 --cpus-per-task=64 torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_id=armtest99 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:24997 \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$LOGDIR \
    wandb_dir=$WANDB_SAVE_DIR \
    wandb_project=longlive-phase-3 \
    max_steps=$MAXSTEPS \
    checkpoint_interval=25 \
    keep_last_n_checkpoints=99 \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_weunz \
    holdout_zarr_list=configs/holdout_rides.txt \
    holdout_eval_root=/projects/u6ex/fbots/frodobots_encoded_weu_holdout \
    holdout_eval_mode=cycle \
    max_ride_frames=900 \
    max_ride_frames_random=true \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    action_dims=[0,1] \
    action_critic_dims=[0,1] \
    denoising_step_list=[1000,625,357.142857,208.333333] \
    max_rolls_per_ride=1 \
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
    stat_anchor_loss_weight=0.0 \
    dmd_asymmetric_scoring_enabled=false \
    dmd_42f_enabled=true \
    dmd_42f_2chunk=false \
    dmd_42f_seed_last=false \
    dmd_42f_gt_after_chunks=2 \
    dmd_42f_rand_sup_slot=false \
    dmd_42f_allsup=false \
    dmd_42f_gt_anchor=true \
    dmd_42f_num_chunks=3 \
    dmd_42f_fix_clean_counterpart=false \
    dmd_42f_clean_drift_enabled=false \
    dmd_42f_clean_match_enabled=false \
    dmd_only_first_chunk_per_ride=true \
    flash_dmd_enabled=false \
    fake_score_ema_weight=0.95 \
    motion_start_threshold=5.0 \
    fake_score_lora_enabled=false \
    gan_enabled=false \
    gan_backbone=ladd_teacher_feat \
    gan_loss_weight=0.0 \
    real_teacher_train_online=false \
    dmd_frozen_teacher_pass_enabled=false \
    real_score_ema_weight=0.0 \
    aux_teacher_loss_weight=0.0 \
    aux_real_clean_x_source=gt \
    aux_noisy_from_raw_gt=true \
    aux_clean_x_random_window=false \
    use_8bit_adam=true \
    gen_gradient_checkpointing=true \
    forward_noiser_enabled=false \
    carn_recurse=false \
    forward_noiser_apply_gt_former=false \
    ladd_adjacent_chunks_enabled=false \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    strict_ode_load=true \
    ode_generator_checkpoint=$ODE_CKPT \
    v14_teacher_checkpoint=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    student_phase_lora_enabled=false \
    $DEXTRA \
    run_name=dmd10k_${DARM}_j${SLURM_JOB_ID}

echo "DMD10K $DARM completed $(date)"
