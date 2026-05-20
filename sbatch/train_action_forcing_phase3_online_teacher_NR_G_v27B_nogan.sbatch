#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-NR-G-v27B_nogan
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_NR_G_v27B
WANDB_SAVE_DIR=wandb

# ---- SAM2 paths (must be set; see PREREQUISITES) ----
GAN_SAM2_CKPT=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_base_plus.pt
GAN_SAM2_CFG=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_b+.yaml

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

# Pre-flight SAM2 sanity checks. Fail-fast on missing prereqs so the
# job exits at submission rather than ~5 min into model load.
python -c "import sam2" 2>/dev/null || {
  echo "ERROR: sam2 package not installed. See PREREQUISITES in this sbatch."
  exit 1
}
[ -f "$GAN_SAM2_CKPT" ] || {
  echo "ERROR: SAM2 checkpoint not found at $GAN_SAM2_CKPT"
  exit 1
}
[ -f "$GAN_SAM2_CFG" ] || {
  echo "ERROR: SAM2 config not found at $GAN_SAM2_CFG"
  exit 1
}

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

echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "GAN_SAM2_CKPT=$GAN_SAM2_CKPT"
echo "GAN_SAM2_CFG=$GAN_SAM2_CFG"
echo "Starting Phase-3 online-teacher NR_G v23 (v22 minus mean anchor, softer std) on $(date)"
echo "  * v23 = v22 with:"
echo "      anti_collapse_loss_weight 1.0 -> 0.2 (softer std floor)"
echo "      anti_collapse_mean_weight 1.0 -> 0.0 (mean anchor OFF)"
echo "  * Std-floor term still fires from step 0 inside _compute_kl_grad"
echo "    when gt_target is available: ReLU(std(GT) - std(pred_x0))^2 (one-sided)."
echo "  * Independent of the alt head — operates on pred_x0 in latent space."
echo "  * v17/v18/v19 collapsed: pairwise (chunk - reference) residual"
echo "    suffered from time-warp blur at AR rollout positions."
echo "  * v20 replaces it with a learned forward-noising ConvNet:"
echo "      f(pred_image) -> 2*pred_image - pred_real"
echo "    At aux-teacher time, ar_residual = noise_aux(gt) - gt."
echo "  * noise_aux warmup_steps=50; before that, AR falls back to no-AR."
echo "  * ~8M params, fp32 ConvNet, AdamW lr=1e-4. Trained every gen step."
echo "  * v15 fast-forwarded by step 16 — DMD weight=0 there, so the"
echo "    teacher is not the cause. Suspects: MANIQA approx, GT-recon, 2x LR."
echo "  * v16 isolates GT-recon: gt_latent_mse_loss_weight 0.1->0,"
echo "    gt_latent_mae_loss_weight 0.05->0. All other v15 knobs unchanged."
echo "  * Prediction: if v16 doesn't fast-forward at step 16, GT-recon was"
echo "    the culprit. If it still does, v17 will test MANIQA / LR."
echo "  * Visual obs from v9-v14:"
echo "      v12 = best (coloured + motion trails); v13 = fast-forward;"
echo "      v14 = blurry/gray; v11 = blue collapse; v9/v10 = slow blur."
echo "  * v15 reverts (a) max_gradient_chunks 2->0, (b) dfake_ratio 2->5,"
echo "    (c) gan_loss_weight 0.3->0.5, (d) aux AR-noise mode combined->off."
echo "  * v15 keeps: CFG off (real_guidance_scale=0), lr=2e-5,"
echo "    real_teacher_lr=7.5e-5, dmd_loss_start_step=40, gt_latent recon."
echo "  * Prediction: if max_gradient_chunks was the dominant cause,"
echo "    v15 looks like v12 with v14's faster convergence."
echo "  * 21-frame iter-1-only student rollout, 250-step max."
echo "  * max_rolls_per_ride=1 — streaming resets every step (no iter k>=2)."
echo "  * Teacher: fresh rank-32 LoRA on v14-merged base, EMA(0.99) target."
echo "  * v13 NEW (vs v10):"
echo "      - CFG off (real_guidance_scale=0)."
echo "      - gan_loss_weight 0.5 -> 0.2 (lighter GAN; AR-noise carries weight)."
echo "      - aux_teacher_ar_noise_mode=combined; AR noise = (chunk-GT) fresh"
echo "        each iter, added to GT before noising (CLEAN-side perturbation)."
echo "        Flow-space scaling α(t)=clamp((1-σ)/σ, max=10), seed-mask ramp"
echo "        over first 1 chunk, 20-step burn-in, EMA-renormalised RMS."
echo "      - max_gradient_chunks=2: only first 2 rolled chunks (frames 0..5)"
echo "        carry gradient for DMD score AND aux teacher. Later chunks are"
echo "        noise-dominated and discarded."
echo "  * gt_latent_mse=0.1, gt_latent_mae=0.05 — direct anti-drift latent"
echo "    L1+L2 against GT, every step, no ramp (carried from v10)."
echo "  * v8 POST-MORTEM: aux_teacher_send_student_grad=true caused deeper"
echo "    collapse (gen_loss -0.36 at step 101 vs v7's +0.15) — kept false here."
echo "  * Stage 1 [0, 20): MANIQA ramps in (full at step 15). LoRA"
echo "    aux trains from step 0 (warmup_steps=80, full at step 80)."
echo "    DMD off, GAN off."
echo "  * Stage 2 [20, 60): DMD engages (start=20, warmup=150, cap=0.5)."
echo "    MANIQA at plateau; LoRA still ramping. GAN still off."
echo "  * Stage 3 [60, end): GAN engages with QUADRATIC ramp"
echo "    (gan_warmup_steps=25, shape=quadratic). D-side starts simul-"
echo "    taneously (no head-start). GAN full at step 85 (plateau 1.0)."
echo "  * Aux LoRA: aux_teacher_send_student_grad=false (LoRA aux is"
echo "    severed from the student; DMD is the sole student-grad path"
echo "    through the teacher). real_teacher_input_source=gt."
echo "    real_score_ema_weight=0.99."
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
    max_steps=250 \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    max_rolls_per_ride=1 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=21 \
    num_workers=0 \
    sample_fps=20 \
    boundary_vae_roundtrip=true \
    local_attn_size_schedule=[[0,21]] \
    dmd_context=self \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=0.0 \
    lr=2e-5 \
    real_teacher_lr=7.5e-5 \
    real_teacher_warmup_steps=0 \
    real_teacher_input_source=gt \
    teacher_cadence=fake \
    aux_teacher_p_schedule_enabled=false \
    aux_teacher_send_student_grad=false \
    aux_teacher_start_step=0 \
    aux_teacher_loss_weight=1.0 \
    aux_teacher_loss_warmup_steps=80 \
    aux_teacher_lora_rank=32 \
    aux_teacher_lora_alpha=64 \
    dmd_loss_weight=0.5 \
    dmd_loss_start_step=20 \
    fake_alt_head_enabled=true \
    fake_alt_head_start_step=0 \
    fake_alt_apply_start_step=0 \
    fake_alt_target_mode=rollout2_student \
    fake_alt_rollout2_num_seed_chunks=2 \
    forward_noiser_enabled=true \
    forward_noiser_hidden_dim=512 \
    forward_noiser_num_blocks=4 \
    forward_noiser_max_carn_step=8 \
    forward_noiser_loss_weight=1.0 \
    forward_noiser_apply_in_aux=true \
    forward_noiser_lr=5e-5 \
    critic_clean_x_source=self \
    aux_real_clean_x_source=causal_AR_GT \
    dmd_normalization_enabled=false \
    dmd_loss_warmup_steps=150 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    real_score_ema_weight=0.0 \
    gan_enabled=false \
    gan_loss_weight=0.2 \
    gan_lr=5e-5 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=768 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=40 \
    gan_disc_start_step=40 \
    gan_warmup_steps=25 \
    gan_warmup_shape=quadratic \
    perceptual_approx_warmup_steps=0 \
    perceptual_approx_ramp_steps=15 \
    gan_updates_per_step=1 \
    gan_r1_gamma=2.0 \
    gan_r2_gamma=2.0 \
    gan_sam2_frame_pool=none \
    gan_sam2_encoder_chunk_size=84 \
    maniqa_approx_loss_weight=1.0 \
    maniqa_n_frames=84 \
    maniqa_n_patches_per_frame=8 \
    maniqa_metric_name=maniqa-pipal \
    gan_critic_dense_distillation=true \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    gt_latent_mse_loss_weight=0.0 \
    gt_latent_mae_loss_weight=0.0 \
    anti_collapse_loss_weight=0.8 \
    anti_collapse_mean_weight=0.0 \
    max_gradient_chunks=5 \
    run_name=action_forcing_phase3_online_teacher_NR_G_v27B_nogan_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
