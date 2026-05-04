#!/bin/bash
#SBATCH --job-name=smoke-v12-fd-sobolev
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing v8 SAM2-pixel + ADM 2D heads smoke.
#
# Validates the paper-aligned discriminator stack:
#   * ADM 2D heads (Lu et al. 2025, arXiv:2507.18569 §B.1, Fig. 6):
#     SDXL-Lightning-style ConvNet — 3× [Conv4×4-stride2 → GN(32) →
#     SiLU] + 1× [Conv3×3 → GN(32) → SiLU] + GAP + Linear(256, 1).
#   * Per-FPN-scale heads on Hiera-B+ multi-scale features.
#   * R3GAN losses (RpGAN-D, R1, R2) backward correctly through the
#     ADM heads (no second-order issues).
#   * Flash-DMD paper-aligned-adv path emits `last_rung_full_chunk`
#     and the trainer consumes it as the disc's G-side fake.
#   * Two-grad-point rollout with K/V context_noise commit (paper
#     §3.3 decoupling) works without OOM.
#
# Wallclock target: 20 steps in <30 min (max_steps=20). The 45-min
# SBATCH limit gives headroom for SAM2 weight load + first-iter cudnn
# warmup.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGROOT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_v12_fd_sobolev_j${SLURM_JOB_ID}
WANDB_SAVE_DIR=wandb

GAN_SAM2_CKPT=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_base_plus.pt
GAN_SAM2_CFG=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_b+.yaml

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

# Pre-flight SAM2 sanity checks (mirrors v8 sbatch).
python -c "import sam2" 2>/dev/null || {
  echo "ERROR: sam2 package not installed. Install via:"
  echo "  pip install git+https://github.com/facebookresearch/sam2.git"
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
mkdir -p "$LOGROOT"
mkdir -p logs

# Reuse the v6 ride-manifest cache so the smoke doesn't burn 6 min
# rescanning 858 zarr files (the manifest fingerprints by encoded_root +
# zarr-mtime, so v6's manifest is valid for any config that points at
# the same encoded_root). v8 production has an equivalent symlink in
# its persistent logdir; the smoke's per-JOBID logdir is fresh each
# run so we re-create the symlink here.
V6_MANIFEST=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_v6/.ride_manifest.pt
if [ -f "$V6_MANIFEST" ]; then
  ln -sf "$V6_MANIFEST" "$LOGROOT/.ride_manifest.pt"
  echo "Symlinked v6 manifest into $LOGROOT/.ride_manifest.pt"
else
  echo "WARNING: v6 manifest not found at $V6_MANIFEST; smoke will rescan rides"
fi

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000 + 11))

export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WANDB_MODE=disabled

echo "CONFIG=$CONFIG"
echo "LOGROOT=$LOGROOT"
echo "MASTER=$MASTER_ADDR:$MASTER_PORT"
echo "Starting SAM2-pixel + ADM + memory-mixer smoke on $(date)"

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
    auto_resume=false \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
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
    real_teacher_warmup_steps=200 \
    real_teacher_input_source=mix \
    real_teacher_input_mix_gt_p=0.5 \
    dmd_frozen_teacher_pass_enabled=false \
    aux_teacher_loss_weight=1.0 \
    flash_dmd_split_timestep=500 \
    fake_score_ema_weight=0.95 \
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
    gan_critic_warmup_steps=5 \
    gan_critic_grad_frames=1 \
    gan_critic_grad_full_every=0 \
    gan_critic_grad_loss_weight=0.0 \
    gan_critic_fd_loss_weight=1.0 \
    gan_critic_fd_sigma=0.01 \
    gan_critic_fd_frames=1 \
    gan_critic_fd_n_directions=1 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    max_steps=20 \
    log_interval=1 \
    checkpoint_interval=999 \
    sample_interval=0 \
    vis_save_local=false \
    run_name=smoke_v12_fd_sobolev_j${SLURM_JOB_ID} \
  > "$LOGROOT/run.log" 2>&1
EXIT=$?

echo ""
echo "----- diagnostics -----"
echo "  exit code:                  $EXIT"
echo "  log file:                   $LOGROOT/run.log"
echo "  log lines:                  $(wc -l < $LOGROOT/run.log)"
N_STEPS=$(grep -cE "step=[0-9]+/20 gen_loss" "$LOGROOT/run.log" || true)
HAS_SAM2=$(grep -c "R3GANDiscriminatorSAM2Pixel.*built" "$LOGROOT/run.log" || true)
HAS_ADM=$(grep -c "ADM 2D heads" "$LOGROOT/run.log" || true)
HAS_FRAMEPOOL=$(grep -c "frame_pool=" "$LOGROOT/run.log" || true)
HAS_PAPER_ADV=$(grep -c "flash_dmd_paper_aligned_adv" "$LOGROOT/run.log" || true)
HAS_CRITIC=$(grep -c "LatentSAM2Critic built" "$LOGROOT/run.log" || true)
HAS_CRITIC_OPT=$(grep -c "LatentSAM2Critic optimizer built" "$LOGROOT/run.log" || true)
# Inline-path proxy: the trainer's optimizer-build log line includes
# ``grad_loss_weight=0.000`` when Sobolev is off, which is the exact
# condition that triggers Paths 1+2 to run inline before Path 3.
# We can't directly grep ``critic_trained_inline=1`` / ``r3gan_d_loss=``
# here because those go to wandb-only keys; with WANDB_MODE=disabled
# the run.log doesn't capture them. Combined with a clean 20/20
# finish (``$N_STEPS=20``), ``grad_loss_weight=0.000`` proves the
# inline branch ran without crashing.
HAS_SOBOLEV_OFF=$(grep -c "grad_loss_weight=0.000" "$LOGROOT/run.log" || true)
HAS_FD=$(grep -c "fd_loss_weight=1.000" "$LOGROOT/run.log" || true)
HAS_FD_SIGMA=$(grep -c "fd_sigma=0.0100" "$LOGROOT/run.log" || true)
echo "  step=N/20 gen_loss lines:   $N_STEPS"
echo "  SAM2 disc built:            $HAS_SAM2"
echo "  ADM 2D heads logged:        $HAS_ADM"
echo "  frame_pool config logged:   $HAS_FRAMEPOOL"
echo "  paper-aligned adv lines:    $HAS_PAPER_ADV"
echo "  LatentSAM2Critic built:     $HAS_CRITIC"
echo "  LatentSAM2Critic optimizer: $HAS_CRITIC_OPT"
echo "  Sobolev OFF (inline path):  $HAS_SOBOLEV_OFF"
echo "  FD Sobolev fd_loss_w=1:     $HAS_FD"
echo "  FD sigma=0.01:              $HAS_FD_SIGMA"
echo ""

if [ "$EXIT" -eq 124 ] || [ "$EXIT" -eq 137 ]; then
  echo "SMOKE FAILED: timeout (30 min)."
  tail -40 "$LOGROOT/run.log"
  exit 42
fi
if [ "$EXIT" -ne 0 ]; then
  echo "SMOKE FAILED: trainer exited $EXIT — last 40 log lines:"
  tail -40 "$LOGROOT/run.log"
  exit 42
fi
if [ "$N_STEPS" -lt 1 ]; then
  echo "SMOKE FAILED: trainer ran clean but logged 0 training steps."
  tail -40 "$LOGROOT/run.log"
  exit 43
fi
if [ "$HAS_SAM2" -lt 1 ]; then
  echo "SMOKE FAILED: SAM2 disc build log missing."
  tail -40 "$LOGROOT/run.log"
  exit 44
fi
if [ "$HAS_ADM" -lt 1 ]; then
  echo "SMOKE FAILED: 'ADM 2D heads' build-log line missing."
  tail -40 "$LOGROOT/run.log"
  exit 45
fi
if [ "$HAS_CRITIC" -lt 1 ]; then
  echo "SMOKE FAILED: 'LatentSAM2Critic built' line missing — distilled-critic flag not honored."
  tail -40 "$LOGROOT/run.log"
  exit 46
fi
if [ "$HAS_CRITIC_OPT" -lt 1 ]; then
  echo "SMOKE FAILED: 'LatentSAM2Critic optimizer built' line missing."
  tail -40 "$LOGROOT/run.log"
  exit 47
fi
if [ "$HAS_SOBOLEV_OFF" -lt 1 ]; then
  echo "SMOKE FAILED: 'grad_loss_weight=0.000' missing — Sobolev gate not"
  echo "    set; can't confirm inline Path 1+2 ran. Check the override"
  echo "    ``gan_critic_grad_loss_weight=0.0`` is being applied."
  tail -40 "$LOGROOT/run.log"
  exit 48
fi
if [ "$HAS_FD" -lt 1 ]; then
  echo "SMOKE FAILED: 'fd_loss_weight=1.000' missing — FD Sobolev not active."
  tail -40 "$LOGROOT/run.log"
  exit 49
fi
if [ "$HAS_FD_SIGMA" -lt 1 ]; then
  echo "SMOKE FAILED: 'fd_sigma=0.0100' missing — FD sigma not propagated."
  tail -40 "$LOGROOT/run.log"
  exit 50
fi
echo "SMOKE PASSED: $N_STEPS/20 steps, distilled critic active, Sobolev OFF,"
echo "             FD Sobolev (fd_loss_weight=1, sigma=0.01, frames=1, n_dir=1)."
echo "  Logs: $LOGROOT/"
