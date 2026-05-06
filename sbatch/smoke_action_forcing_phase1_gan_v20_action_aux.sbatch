#!/bin/bash
#SBATCH --job-name=smoke-v20-action-aux
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Smoke for v20: validates the action-loss aux additions on top of the
# v18-style stable baseline.
#   * action_teacher_mode=all              (CoTracker + ss_vae loaded)
#   * action_critic_aux_enabled=true       (gen-side z-guidance)
#   * state_probe_aux_enabled=true         (LoRA-side state_probe loss)
#   * lora_action_critic_z_guidance_weight (LoRA-side z-guidance)
# Smoke target: 20 steps in <30 min.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGROOT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_v20_action_aux_j${SLURM_JOB_ID}
WANDB_SAVE_DIR=wandb

GAN_SAM2_CKPT=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_base_plus.pt
GAN_SAM2_CFG=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_b+.yaml

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

python -c "import sam2" 2>/dev/null || {
  echo "ERROR: sam2 package not installed."
  exit 1
}
[ -f "$GAN_SAM2_CKPT" ] || { echo "ERROR: SAM2 ckpt missing"; exit 1; }
[ -f "$GAN_SAM2_CFG" ]  || { echo "ERROR: SAM2 cfg missing";  exit 1; }

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"
mkdir -p "$LOGROOT"
mkdir -p logs

# Reuse v6's ride manifest cache so smoke doesn't rescan rides.
V6_MANIFEST=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_v6/.ride_manifest.pt
if [ -f "$V6_MANIFEST" ]; then
  ln -sf "$V6_MANIFEST" "$LOGROOT/.ride_manifest.pt"
  echo "Symlinked v6 manifest into $LOGROOT/.ride_manifest.pt"
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000 + 11))

export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=online
export WANDB_PROJECT=arrwm
export WANDB_RUN_NAME=smoke_v20_action_aux_${SLURM_JOB_ID}

echo "CONFIG=$CONFIG"
echo "LOGROOT=$LOGROOT"
echo "MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "Starting v20 action-aux smoke on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true

timeout --kill-after=30 1800 \
srun torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_smk \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$LOGROOT \
    wandb_dir=$WANDB_SAVE_DIR \
    wandb_project=longlive-phase1-smoke \
    auto_resume=false \
    action_teacher_mode=all \
    action_critic_aux_enabled=true \
    state_probe_aux_enabled=false \
    memory_audit_enabled=true \
    generator_action_z_guidance_weight=0.125 \
    lora_action_critic_z_guidance_weight=0.0625 \
    max_rolls_per_ride=300 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=42 \
    num_workers=0 \
    boundary_vae_roundtrip=false \
    collapse_mae_threshold=0.2 \
    mae_extension_threshold=0.2 \
    local_attn_size_schedule=[[0,42]] \
    dmd_context=mix \
    dmd_context_mix_p=0.15 \
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
    perceptual_approx_warmup_steps=3 \
    perceptual_approx_mean_align_weight=0.1 \
    perceptual_approx_n_interp_samples=3 \
    gan_critic_n_interp_samples=3 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=512 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=3 \
    gan_critic_grad_frames=1 \
    gan_critic_grad_full_every=0 \
    gan_critic_grad_loss_weight=0.0 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    max_steps=20 \
    log_interval=1 \
    checkpoint_interval=999 \
    sample_interval=0 \
    vis_save_local=false \
    run_name=smoke_v20_action_aux_j${SLURM_JOB_ID} \
  > "$LOGROOT/run.log" 2>&1
EXIT=$?

echo ""
echo "----- diagnostics -----"
echo "  exit code:  $EXIT"
echo "  log file:   $LOGROOT/run.log"
N_STEPS=$(grep -cE "step=[0-9]+/20 gen_loss" "$LOGROOT/run.log" || true)
HAS_AC_BUILT=$(grep -c "action_critic optimizer\|action_critic load:" "$LOGROOT/run.log" || true)
HAS_SP_BUILT=$(grep -c "state_probe optimizer built" "$LOGROOT/run.log" || true)
HAS_SP_ATTACH=$(grep -c "state_probe attached to real_score" "$LOGROOT/run.log" || true)
HAS_TEACHER=$(grep -c "Action teacher.*built\|cotracker\|ss_vae" "$LOGROOT/run.log" || true)
echo "  step=N/20:                     $N_STEPS"
echo "  action_critic build/load:      $HAS_AC_BUILT"
echo "  state_probe optimizer built:   $HAS_SP_BUILT"
echo "  state_probe attached real_score: $HAS_SP_ATTACH"
echo "  action teacher (CoTracker+ssvae): $HAS_TEACHER"
echo ""

if [ "$EXIT" -ne 0 ]; then
  echo "SMOKE FAILED: trainer exited $EXIT — last 60 log lines:"
  tail -60 "$LOGROOT/run.log"
  exit 42
fi
if [ "$N_STEPS" -lt 1 ]; then
  echo "SMOKE FAILED: trainer ran clean but logged 0 training steps."
  tail -60 "$LOGROOT/run.log"
  exit 43
fi
echo "SMOKE PASSED: $N_STEPS/20 steps, action+state_probe heads active."
echo "  Logs: $LOGROOT/"
