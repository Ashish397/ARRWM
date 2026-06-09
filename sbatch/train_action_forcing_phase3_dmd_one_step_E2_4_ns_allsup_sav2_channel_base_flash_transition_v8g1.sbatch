#!/bin/bash
#SBATCH --job-name=af-p3-cb-flashtr-v8g1
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
# Phase-3 DMD one-step — v8g1 = v8e8 + AUX TEACHER TRAINING (online LoRA
# teacher that learns to remove CARN drift), the only delta from e8.
#
# ---- What g1 turns on (vs e8) ----
# The frozen merged-v14 teacher becomes an ONLINE LoRA teacher trained by
# an auxiliary denoising pass, while the DMD generator loss is scored
# against an EMA target-network copy of that LoRA (stable distribution
# target, memory-cheap — no full deepcopy).
#
# Aux pass (model/dmd_action_forcing.py:_compute_aux_teacher_loss_streaming):
#   * noisy_x = GT for ALL 7 chunks (not the 3/6-chunk mix the DMD step
#     uses). clean_x = GT, as now.
#   * Progressive CARN applied to the GT noisy_x, LOCAL-INDEX ladder
#     (aux_carn_level_mode=local_index): chunk 0 untouched (0 CARN),
#     chunk 1 -> 1 application, chunk 2 -> 2, ... chunk 6 -> 6 — applied
#     RECURSIVELY (output fed back), 6 application steps across 7 chunks.
#     Same ForwardNoiser model as the GAN former, STEP-UNCONDITIONED
#     (forward_noiser_step_unconditioned=true -> carn_step=0 every apply;
#     progressiveness comes from the per-chunk recursion depth, not the
#     FiLM step conditioning). NOTE: the aux path applies CARN RAW (no
#     moment-preserving restore) — intentional, so the teacher learns to
#     undo the FULL drift incl. brightness (delta 1 left raw; the GAN
#     former still gets the texture-only moment restore).
#   * Teacher denoises add_noise(progressive-CARN GT, eps, t) back to
#     TRUE GT -> teaches the teacher to remove CARN.
#
# Stable DMD target: real_score_ema_weight=0.95 (EMA of the LoRA). The
# DMD real-score query swaps to the EMA copy, so the distribution target
# the student matches stays stable while the live LoRA trains. Cheaper
# than dmd_frozen_teacher_pass (deepcopy), which stays OFF.
#
# MEMORY: aux adds ~+8-13 GB (one extra teacher fwd+bwd w/ retained
# activations; LoRA optimizer/grad are negligible, CARN is no_grad). On a
# CLEAN 95 GiB card vs e8's ~67 GiB this lands ~75-80 GiB — feasible but
# tight. If it OOMs, the levers (in order): gan_updates_per_step 5->3,
# ladd_pairs_per_step 7->5, or max_ride_frames 900->600. Do NOT cut the
# aux pass or CARN.
#
# ---- e8 base (unchanged below) ----
# e8 = e6 with mean_equalize OFF (GAN owns brightness), gan_loss_weight
# 1.5, random 900-frame windows, rel_tol 0.05. Stat anchor
# target_matching (whole-pool exhaustive, k=4). CARN-former texture-only
# (moment-restore), GAN at 5/7 strength, R1-only, flash-DMD 42f allsup.
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
echo "Starting Phase-3 DMD ONE-STEP (v8g1 = e8 + aux online-LoRA teacher de-CARN) on $(date)"
echo "  * allsup: 3 supervised chunks; cf=18 seed + GT-anchor = 7 GT; local_attn=24."
echo "  * 1 roll per ride (fresh seed each step)."
echo "  * GAN: ON (ladd_teacher_feat, matched gt_transition), R1-only, v6e strength (5/7)."
echo "  * CARN-former: learned FN, +1 cartoon shift (latent-space sliced-Wasserstein teacher-feat), rollout2 ON."
echo "  * AUX TEACHER: online LoRA, denoise progressive-CARN(GT)->GT, local_index ladder 0..6; DMD scored vs EMA LoRA (0.95)."
echo "  * dmd_normalization: ON, denom floor=1.0 -> divide by max(|f|,1)."
echo "  * MAE gate: ON (r_full=2.0, ema=0.0 [per-sample, no smoothing], min_weight=0.0, exponent=0.75)."
echo "  * Teacher: ONLINE LoRA (aux de-CARN) + EMA target for DMD scoring."
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
    stat_anchor_loss_weight=1.0 \
    stat_anchor_STD_short_weight=0.0 \
    stat_anchor_STD_long_weight=0.0 \
    stat_anchor_mode=target_matching \
    stat_anchor_match_k=4 \
    stat_anchor_M2_short_weight=0.1 \
    stat_anchor_M2_long_weight=0.1 \
    stat_anchor_SOS_short_weight=0.0 \
    stat_anchor_SOS_long_weight=0.0 \
    stat_anchor_M1_short_weight=3e-9 \
    stat_anchor_M1_long_weight=3e-9 \
    stat_anchor_TV_short_weight=0.1 \
    stat_anchor_TV_long_weight=0.1 \
    stat_anchor_rel_tol_short=0.05 \
    stat_anchor_rel_tol_long=0.05 \
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
    fake_score_lora_enabled=true \
    fake_score_lora_rank=32 \
    fake_score_lora_alpha=64 \
    fake_score_lora_dropout=0.0 \
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
    ladd_gt_transition_mean_equalize=false \
    ladd_gt_transition_n_real=0 \
    ladd_gt_transition_match=true \
    ladd_gt_transition_match_k=4 \
    forward_noiser_enabled=true \
    forward_noiser_loss_mode=teacher_feat \
    forward_noiser_step_unconditioned=true \
    forward_noiser_feat_t=0 \
    forward_noiser_sw_n_proj=64 \
    forward_noiser_sw_max_tokens=4096 \
    fake_alt_rollout2_num_seed_chunks=5 \
    ladd_adjacent_chunks_enabled=false \
    ladd_adjacent_chunks_weight=0.0 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    gen_gradient_checkpointing=true \
    real_teacher_train_online=true \
    aux_teacher_lora_rank=32 \
    aux_teacher_lora_alpha=64 \
    real_score_ema_weight=0.95 \
    dmd_frozen_teacher_pass_enabled=false \
    aux_teacher_loss_weight=1.0 \
    aux_teacher_start_step=0 \
    aux_teacher_loss_warmup_steps=20 \
    real_teacher_lr=5e-5 \
    real_teacher_warmup_steps=20 \
    forward_noiser_apply_in_aux=true \
    forward_noiser_apply_strategy=learned \
    aux_real_clean_x_source=gt \
    aux_carn_level_mode=local_index \
    run_name=2_4_ns_allsup_sav2_channel_base_flash_transition_v8g1_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
