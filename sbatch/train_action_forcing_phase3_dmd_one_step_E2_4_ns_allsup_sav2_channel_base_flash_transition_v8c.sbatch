#!/bin/bash
#SBATCH --job-name=af-p3-cb-flashtr-v8c
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-3 DMD one-step — v8c = v6e + bring TRAINING context up to the
# INFERENCE (7-chunk eval) operating regime.
#
# Two coupled changes vs v6e (so the supervised chunks roll from clean
# GT context that matches deployment, not a degraded training-only one):
#   * dmd_42f_gt_anchor=true  (B): the +npb leading anchor is now a CLEAN
#     GT chunk (seed-prefilled) instead of a drifted student rollout —
#     removes the generate-vs-score mismatch (student anchor in the KV
#     slot the 42f teacher scores with GT) AND adds a 7th clean GT chunk
#     on top of the 6-chunk (cf=18) seed, so the supervised chunks see
#     pure GT context.
#   * local_attn_size=24            : 8-chunk attention window (7 GT
#     context + self) so that 7th GT chunk is actually ATTENDED (the
#     21-frame window only held 6 context chunks). The 7-chunk eval runs
#     on the same pipeline, so it ALSO uses window 24 -> train/eval match.
#
# NOTE: cf stays 18 (Option "C", cf=21, is BOTH blocked by the hard guard
# cf < num_training_frames=21 AND redundant — the GT anchor already adds
# the 7th GT chunk; a cf=21 8th chunk would just evict from the 24-window).
#
# RESEARCH RISK (not a bug): the generator is warm-started from an ODE
# checkpoint trained at local_attn_size=21; attending to 24 is a wider
# (slightly OOD) window the student may need a few steps to adapt to.
#
# Everything else identical to v6e (R1-only GAN, matched gt_transition).
# If validated, gt_anchor + local_attn=24 become defaults.
# 21->24 frame window, 200-step max.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase3_dmd.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_dmd_one_step
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

# GAN OFF -> no SAM2 prereq checks needed.

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
echo "Starting Phase-3 DMD ONE-STEP v8c (v6e + GT-anchor + local_attn=24) on $(date)"
echo "  * allsup: 3 supervised chunks per 7-chunk roll."
echo "  * cf=18 seed + clean GT anchor = 7 GT chunks; local_attn_size=24."
echo "  * 1 roll per ride (fresh seed each step)."
echo "  * GAN: ON (ladd_teacher_feat, matched gt_transition, R1-only)."
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
    max_steps=200 \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
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
    stat_anchor_loss_weight=1.0 \
    stat_anchor_STD_short_weight=0.0 \
    stat_anchor_STD_long_weight=0.0 \
    stat_anchor_M2_short_weight=0.1 \
    stat_anchor_M2_long_weight=0.1 \
    stat_anchor_SOS_short_weight=0.0 \
    stat_anchor_SOS_long_weight=0.0 \
    stat_anchor_M1_short_weight=3e-9 \
    stat_anchor_M1_long_weight=3e-9 \
    stat_anchor_TV_short_weight=0.1 \
    stat_anchor_TV_long_weight=0.1 \
    stat_anchor_rel_tol_short=0.1 \
    stat_anchor_rel_tol_long=0.1 \
    dmd_asymmetric_scoring_enabled=false \
    dmd_42f_enabled=true \
    dmd_42f_2chunk=true \
    dmd_42f_seed_last=false \
    dmd_42f_gt_after_chunks=3 \
    dmd_42f_rand_sup_slot=false \
    dmd_42f_allsup=true \
    dmd_42f_gt_anchor=true \
    dmd_only_first_chunk_per_ride=true \
    flash_dmd_enabled=true \
    fake_score_ema_weight=0.95 \
    gan_enabled=true \
    gan_backbone=ladd_teacher_feat \
    gan_loss_weight=1.0 \
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
    ladd_pairs_per_step=7 \
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
    ladd_gt_transition_n_real=0 \
    ladd_gt_transition_match=true \
    ladd_gt_transition_match_k=4 \
    ladd_adjacent_chunks_enabled=false \
    ladd_adjacent_chunks_weight=0.0 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=2_4_ns_allsup_sav2_channel_base_flash_transition_v8c_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
