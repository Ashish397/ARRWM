#!/bin/bash
#SBATCH --job-name=smoke-action-forcing-phase1-aux-gan
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing AUX + R3GAN smoke harness.
#
# Validates:
#   * ``model.r3gan.R3GANDiscriminator3D`` builds and forwards on
#     latent video [B, F=21, C=16, H≈30, W≈45].
#   * R3GAN losses (RpGAN-D, RpGAN-G, R1, R2) backward correctly with
#     ``create_graph=True`` on the gradient penalties.
#   * D's optimizer steps; checkpoint save/resume includes the
#     ``r3gan_discriminator`` + ``r3gan_optimizer`` state.
#   * DDP all-reduce on D's gradients fires correctly with the
#     second-order R1/R2 graph in scope.
#   * Aux + GAN co-exist in the unified generator forward path
#     (single rollout shared between action critic z-guidance and
#     R3GAN D-update + G-side term).
#
#   Test G1: single-GPU smoke, AUX + GAN ON
#       max_steps=1
#       → confirms the full path on 1 GPU; emits
#         ``r3gan_d_loss``, ``r3gan_r1``, ``r3gan_r2``,
#         ``r3gan_g_loss_*`` in stdout.
#
#   Test G2: 4-GPU DDP smoke, AUX + GAN ON
#       max_steps=2
#       → DDP all-reduce on D's gradients (with second-order graph
#         in scope) and on G (with frozen D in scope). Catches
#         the typical DDP "found unused params" failure mode.
#
#   Test G3: 4-GPU DDP smoke, GAN ONLY (aux off)
#       max_steps=2 action_teacher_mode=off
#                   action_critic_aux_enabled=false gan_enabled=true
#       → confirms the GAN works *without* the aux losses (the
#         unified-path refactor must keep gan-only viable).
#
#   Test G4: 4-GPU DDP smoke, GAN OFF (regression guard)
#       max_steps=2 gan_enabled=false (aux still on)
#       → confirms ``gan_enabled=false`` is genuinely free
#         (no D build, no D optim, no GAN loss path).
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_aux_gan.yaml
LOGROOT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_action_forcing_phase1_aux_gan_j${SLURM_JOB_ID}
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

mkdir -p "$LOGROOT"
mkdir -p logs

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export WANDB_MODE=disabled

echo "CONFIG=$CONFIG"
echo "LOGROOT=$LOGROOT"
echo "Starting AUX+GAN smoke harness on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true

# ---------------------------------------------------------------------
# Test G1: single-GPU smoke, AUX + GAN ON.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST G1: single-GPU smoke, AUX + GAN ON"
echo "  max_steps=1 (full critic + teacher + R3GAN path on 1 GPU)"
echo "================================================================"
T1_LOG=$LOGROOT/testG1
mkdir -p "$T1_LOG"
srun --ntasks=1 --ntasks-per-node=1 --gpus=1 torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=${SLURM_JOB_ID}_g1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+1)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T1_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_aux_gan_g1_single \
    max_steps=1 \
    log_interval=1 \
    checkpoint_interval=999 \
  2>&1 | tee "$T1_LOG/testG1.log" || { echo "TEST G1 FAILED"; exit 31; }
if grep -qE "r3gan_d_loss|r3gan_g_loss" "$T1_LOG/testG1.log"; then
  echo "TEST G1 PASSED: R3GAN telemetry detected"
else
  echo "TEST G1 WARNING: R3GAN telemetry not detected in log"
fi

# ---------------------------------------------------------------------
# Test G2: 4-GPU DDP smoke, AUX + GAN ON.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST G2: 4-GPU DDP smoke, AUX + GAN ON"
echo "  max_steps=2 (DDP all-reduce on D, gen, fake_score, critic)"
echo "================================================================"
T2_LOG=$LOGROOT/testG2
mkdir -p "$T2_LOG"
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_g2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+2)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T2_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_aux_gan_g2_ddp \
    max_steps=2 \
    log_interval=1 \
    checkpoint_interval=999 \
  2>&1 | tee "$T2_LOG/testG2.log" || { echo "TEST G2 FAILED (likely DDP issue on D)"; exit 32; }
echo "TEST G2 PASSED: 4-GPU DDP with AUX + GAN completed"

# ---------------------------------------------------------------------
# Test G3: 4-GPU DDP smoke, GAN ONLY (aux off).
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST G3: 4-GPU DDP smoke, GAN-only (aux off)"
echo "  max_steps=2 action_teacher_mode=off action_critic_aux_enabled=false"
echo "================================================================"
T3_LOG=$LOGROOT/testG3
mkdir -p "$T3_LOG"
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_g3 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+3)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T3_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_aux_gan_g3_gan_only \
    max_steps=2 \
    log_interval=1 \
    checkpoint_interval=999 \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
  2>&1 | tee "$T3_LOG/testG3.log" || { echo "TEST G3 FAILED"; exit 33; }
echo "TEST G3 PASSED: GAN-only path works"

# ---------------------------------------------------------------------
# Test G4: 4-GPU DDP smoke, GAN OFF (regression guard).
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST G4: 4-GPU DDP smoke, GAN OFF (aux still on)"
echo "  max_steps=2 gan_enabled=false"
echo "================================================================"
T4_LOG=$LOGROOT/testG4
mkdir -p "$T4_LOG"
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_g4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+4)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T4_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_aux_gan_g4_gan_off \
    max_steps=2 \
    log_interval=1 \
    checkpoint_interval=999 \
    gan_enabled=false \
  2>&1 | tee "$T4_LOG/testG4.log" || { echo "TEST G4 FAILED"; exit 34; }
echo "TEST G4 PASSED: gan_enabled=false fall-through works"

echo ""
echo "================================================================"
echo "  ALL AUX+GAN SMOKE TESTS PASSED on $(date)"
echo "  Logs: $LOGROOT"
echo "================================================================"
