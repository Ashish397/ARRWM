#!/bin/bash
#SBATCH --job-name=af-all-freal-cd
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
# ======================= af-all-freal-cd (FULL) ======================
# Distill the RAW v14d 40-step checkpoint with the new flag-gated
# features (af-all IQA depth-gate + online causal-CD loss), with a
# FROZEN v14d teacher (freal). 8 nodes, 3h wall.
#
# Base recipe = canonical af_roll rolling DMD
#   (sbatch/train_arrwm_p4_af_roll_wave.sbatch,
#    CONFIG=configs/action_forcing_phase3_dmd.yaml). Copied, not edited.
#
# Deltas vs the af_roll base:
#   * FRESH v14d init (NOT resuming a prior phase1_step / stat_wave
#     student): the af_roll phase1_step pre-stage block is BYPASSED and
#     auto_resume=false, so the student/teacher init from the v14d ckpt.
#       - student generator <- ode_generator_checkpoint=$V14D_CKPT
#       - frozen teacher    <- v14_teacher_checkpoint=$V14D_CKPT
#     (See the KNOWN RISK block below re: the student-generator key.)
#   * FREAL: frozen teacher. real_teacher_train_online=false (base=true),
#     real_score_ema_weight=0.0 (base=0.95), aux_teacher_loss_weight=0.0,
#     real_teacher_anchor_lambda=0.0, real_guidance_scale=0.0.
#   * GAN OFF: gan_enabled=false (base=true).
#   * Flash-DMD OFF: flash_dmd_enabled=false (base=true).
#   * CD loss ON: cd_loss_enabled=true, cd_loss_weight=1.0,
#     cd_ema_decay=0.99, cd_loss_guidance_scale=0.0,
#     cd_loss_warmup_steps=50 (FULL: 50-step ramp).
#   * af-all IQA depth-gate ON: rolling_toothpaste_enabled=true (the
#     gate is nested in the toothpaste init block) + af_all_iqa_*.
#
# >>> KNOWN RISK / OPEN ITEM (read before submit) <<<
# The af/DMD trainer's STUDENT-generator loader
# (_load_generator_from_ode_checkpoint, model/dmd_action_forcing.py:3081)
# reads ONLY the "generator" key from ode_generator_checkpoint and
# raises KeyError if absent (line 3096-3099). The v14d checkpoint
# (causal_lora_step0002150.pt) is a LoRA-ONLY checkpoint: it has a
# "lora" key but NO "generator" key. The lora->base merge idiom for a
# student generator lives only in the SEPARATE ODE trainer
# (action-forcing/af_model/ode_regression.py). So there is currently NO
# config-only way to fresh-init the af/DMD student from the raw v14d
# lora ckpt. The TEACHER (real_score) DOES load v14d cleanly via
# v14_teacher_checkpoint (reads the "lora" key + merges).
# Resolution options (pick one before running):
#   (A) small code change to _load_generator_from_ode_checkpoint to
#       accept a "lora"-only ckpt and merge it into the generator base
#       (mirror of _load_real_score_with_v14_lora), then point
#       ode_generator_checkpoint at the v14d ckpt; OR
#   (B) run ODE-distill-F first to produce a "generator"-keyed v14d
#       student ckpt, then point ode_generator_checkpoint at THAT.
# Until then, the ode_generator_checkpoint override below points at the
# v14d ckpt and WILL KeyError on load. Left wired-to-v14d on purpose so
# the intended init is explicit for review.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase3_dmd.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_dmd_one_step
WANDB_SAVE_DIR=wandb

# v14d RAW 40-step checkpoint (LoRA-only: has 'lora', no 'generator').
V14D_CKPT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14d_critic8_natural/causal_lora_step0002150.pt

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
echo "Starting af-all-freal-cd FULL (fresh v14d init, FROZEN teacher, CD loss + af-all IQA gate) on $(date)"
echo "  * FRESH v14d init: phase1_step pre-stage BYPASSED, auto_resume=false."
echo "  * Teacher: FROZEN merged-v14d (online=false, ema=0.0, aux=0.0, anchor=0.0, rg=0.0)."
echo "  * GAN OFF, Flash-DMD OFF."
echo "  * CD loss ON (weight=1.0, ema=0.99, guidance=0.0, warmup=50)."
echo "  * af-all IQA depth-gate ON (toothpaste rolling kept on)."
echo "  * v14d ckpt: $V14D_CKPT"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# NOTE: the af_roll phase1_step pre-stage block is INTENTIONALLY OMITTED
# here. This is a FRESH v14d start, so we do NOT copy a prior rolling /
# stat_wave student into the run dir and we set auto_resume=false. The
# student is meant to init from the v14d generator_ckpt instead (see the
# KNOWN RISK block in the header re: the generator-key blocker).
RUN_NAME=af_all_freal_cd_j${SLURM_JOB_ID}
RUN_DIR=$LOGDIR/$RUN_NAME
mkdir -p "$RUN_DIR"

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
    max_steps=500 \
    checkpoint_interval=100 \
    empty_cache_interval=5 \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_weu \
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
    max_rolls_per_ride=32 \
    num_chunks_roll_forward=2 \
    streaming_max_length=100 \
    streaming_min_new_frame=6 \
    dmd_42f_rolling_sup_new=true \
    auto_resume=false \
    ode_generator_checkpoint=$V14D_CKPT \
    v14_teacher_checkpoint=$V14D_CKPT \
    strict_ode_load=false \
    ftv3_postbuild_enabled=true \
    fn_frontier_pairs=true \
    streaming_mae_going_threshold=0.7 \
    streaming_mae_collapse_threshold=0.9 \
    ftv3_going_min_depth=3 \
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
    dmd_42f_2chunk=false \
    dmd_42f_seed_last=false \
    dmd_42f_gt_after_chunks=2 \
    dmd_42f_rand_sup_slot=false \
    dmd_42f_allsup=false \
    dmd_42f_gt_anchor=true \
    dmd_42f_num_chunks=3 \
    dmd_42f_fix_clean_counterpart=false \
    dmd_42f_clean_drift_enabled=true \
    dmd_42f_clean_drift_start_step=0 \
    dmd_42f_clean_drift_full_step=0 \
    dmd_42f_clean_drift_couple_rope=true \
    dmd_only_first_chunk_per_ride=false \
    flash_dmd_enabled=false \
    fake_score_ema_weight=0.95 \
    motion_start_threshold=5.0 \
    fake_score_lora_enabled=false \
    gan_enabled=false \
    gan_backbone=ladd_teacher_feat \
    rolling_toothpaste_enabled=true \
    af_all_iqa_gate_enabled=true \
    af_all_iqa_musiq_tol=5.0 \
    af_all_iqa_niqe_tol=0.5 \
    af_all_iqa_late_early_drop=5.0 \
    af_all_iqa_ema_decay=0.9 \
    af_all_iqa_agree_threshold=0.5 \
    cd_loss_enabled=true \
    cd_loss_weight=1.0 \
    cd_ema_decay=0.99 \
    cd_loss_guidance_scale=0.0 \
    cd_loss_warmup_steps=50 \
    real_teacher_train_online=false \
    dmd_frozen_teacher_pass_enabled=false \
    real_score_ema_weight=0.0 \
    real_teacher_anchor_lambda=0.0 \
    aux_teacher_loss_weight=0.0 \
    use_8bit_adam=true \
    gen_gradient_checkpointing=true \
    forward_noiser_enabled=true \
    forward_noiser_loss_mode=teacher_feat \
    forward_noiser_step_unconditioned=false \
    carn_recurse=true \
    forward_noiser_feat_t=0 \
    forward_noiser_sw_n_proj=64 \
    forward_noiser_sw_max_tokens=4096 \
    fake_alt_rollout2_num_seed_chunks=4 \
    forward_noiser_apply_gt_former=true \
    forward_noiser_apply_gt_level=1 \
    forward_noiser_apply_gt_drift_cap=true \
    ladd_adjacent_chunks_enabled=false \
    ladd_adjacent_chunks_weight=0.0 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=af_all_freal_cd_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
