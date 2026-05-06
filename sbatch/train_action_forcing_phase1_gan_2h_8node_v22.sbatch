#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-gan-v22
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
# v22: sensor-gated dmd_context_mix_p (no linear ramp).
#   * dmd_context_mix_p starts at 0.15 (same as v21 start).
#   * Ramping disabled (ramp_steps=0).
#   * Sensor gate enabled: reads the previous gen-iter's
#     ``gen/dmd_pf_minus_pr_mae`` (= |pred_fake - pred_real|, the
#     unnormalised DMD push direction). When it drops below 0.1
#     the scorers are converging — student is catching up to teacher
#     and the LoRA needs more GT support to remain a useful target.
#     Sensor flips mix_p 0.15 -> 0.7. When the metric rises back
#     above 0.1, flips back to 0.15. Stateless single-threshold gate
#     (no hysteresis); reads previous gen-iter value (1-iter lag).
#
# Action-loss aux stays DISABLED (same as v21).
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_v22
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "GAN_SAM2_CKPT=$GAN_SAM2_CKPT"
echo "GAN_SAM2_CFG=$GAN_SAM2_CFG"
echo "Starting Phase-1 Action-Forcing DMD + R3GAN v22 on $(date)"
echo "v22 from v21:"
echo "  * dmd_context_mix_p sensor-gated (no ramp)."
echo "  * Threshold: gen/dmd_pf_minus_pr_mae < 0.1 -> mix_p=0.7."
echo "    Above 0.1 -> mix_p=0.15."
echo "  * Action-loss aux still DISABLED."


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
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    max_rolls_per_ride=300 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=42 \
    num_workers=0 \
    boundary_vae_roundtrip=true \
    collapse_mae_threshold=0.2 \
    mae_extension_threshold=0.2 \
    local_attn_size_schedule=[[0,42]] \
    dmd_context=mix \
    dmd_context_mix_p=0.15 \
    dmd_context_mix_p_target=0.15 \
    dmd_context_mix_p_ramp_steps=0 \
    dmd_context_mix_p_sensor_enabled=true \
    dmd_context_mix_p_sensor_threshold=0.1 \
    dmd_context_mix_p_sensor_low=0.15 \
    dmd_context_mix_p_sensor_high=0.7 \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=1.0 \
    real_teacher_lr=5e-5 \
    real_teacher_warmup_steps=30 \
    real_teacher_input_source=mix \
    real_teacher_input_mix_gt_p=0.7 \
    dmd_frozen_teacher_pass_enabled=false \
    aux_teacher_loss_weight=1.0 \
    flash_dmd_split_timestep=500 \
    fake_score_ema_weight=0.99 \
    flash_dmd_paper_aligned_adv=true \
    gan_loss_weight=1.0 \
    gan_warmup_steps=400 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=512 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=25 \
    gan_critic_grad_frames=1 \
    gan_critic_grad_full_every=0 \
    gan_critic_grad_loss_weight=0.0 \
    gan_sam2_frame_pool=none \
    gan_sam2_encoder_chunk_size=21 \
    lpips_loss_weight=0.0 \
    pixel_recon_loss_weight=0.0 \
    mse_approx_loss_weight=0.0 \
    mse_approx_l2_weight=1.0 \
    mse_approx_l1_weight=0.5 \
    lpips_approx_loss_weight=1.0 \
    msssim_approx_loss_weight=1.0 \
    msssim_target_domain=edges \
    perceptual_approx_d_model=256 \
    perceptual_approx_num_blocks=4 \
    perceptual_approx_warmup_steps=25 \
    perceptual_approx_mean_align_weight=0.1 \
    perceptual_approx_n_interp_samples=0 \
    gan_critic_n_interp_samples=0 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=action_forcing_phase1_gan_j${SLURM_JOB_ID}_v22

echo "Training completed on $(date)"
