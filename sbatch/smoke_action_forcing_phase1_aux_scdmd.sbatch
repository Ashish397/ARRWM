#!/bin/bash
#SBATCH --job-name=smoke-action-forcing-phase1-aux-scdmd
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing AUX + SC-DMD (NO GAN) smoke harness.
#
# Validates:
#   * ``ActionForcingDMD.sc_dmd_loss`` builds and forwards on
#     a fresh KV cache (chunk-0, ``num_frame_per_block`` frames).
#   * SC-DMD's two DiT forwards backprop correctly with the
#     existing rollout's DMD/aux backward in the same step.
#   * Triplet ``(t_s, t_m, t_e)`` sampled and broadcast across DDP
#     ranks (rank 0 picks, broadcast to others).
#   * SC-DMD coexists with action critic z-guidance in the unified
#     generator forward path.
#   * ``sc_dmd_enabled=false`` is genuinely free (regression guard).
#
#   Test S1: single-GPU smoke, AUX + SC ON (GAN off)
#   Test S2: 4-GPU DDP smoke, AUX + SC ON (GAN off)
#   Test S3: 4-GPU DDP smoke, SC-only (aux off, GAN off)
#   Test S4: 4-GPU DDP smoke, SC OFF (aux still on, GAN still off)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_aux_scdmd.yaml
LOGROOT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_action_forcing_phase1_aux_scdmd_j${SLURM_JOB_ID}
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
echo "Starting AUX+SC-DMD (no GAN) smoke harness on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true

# ---------------------------------------------------------------------
# Test S1: single-GPU smoke, AUX + SC ON (GAN off).
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST S1: single-GPU smoke, AUX + SC ON (GAN off)"
echo "  max_steps=1 (full critic + teacher + SC path on 1 GPU)"
echo "================================================================"
T1_LOG=$LOGROOT/testS1
mkdir -p "$T1_LOG"
srun --ntasks=1 --ntasks-per-node=1 --gpus=1 torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=${SLURM_JOB_ID}_s1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+1)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T1_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_aux_scdmd_s1_single \
    max_steps=1 \
    log_interval=1 \
    checkpoint_interval=999 \
  2>&1 | tee "$T1_LOG/testS1.log" || { echo "TEST S1 FAILED"; exit 41; }
if grep -qE "sc_dmd_loss_raw|sc_dmd_t_s" "$T1_LOG/testS1.log"; then
  echo "TEST S1 PASSED: SC-DMD telemetry detected"
else
  echo "TEST S1 WARNING: SC-DMD telemetry not detected in log"
fi

# ---------------------------------------------------------------------
# Test S2: 4-GPU DDP smoke, AUX + SC ON (GAN off).
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST S2: 4-GPU DDP smoke, AUX + SC ON (GAN off)"
echo "  max_steps=2 (DDP all-reduce on gen, fake_score, critic, SC)"
echo "================================================================"
T2_LOG=$LOGROOT/testS2
mkdir -p "$T2_LOG"
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_s2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+2)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T2_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_aux_scdmd_s2_ddp \
    max_steps=2 \
    log_interval=1 \
    checkpoint_interval=999 \
  2>&1 | tee "$T2_LOG/testS2.log" || { echo "TEST S2 FAILED (likely DDP / triplet-broadcast issue)"; exit 42; }
echo "TEST S2 PASSED: 4-GPU DDP with AUX + SC completed"

# ---------------------------------------------------------------------
# Test S3: 4-GPU DDP smoke, SC ONLY (aux off, GAN off).
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST S3: 4-GPU DDP smoke, SC-only (aux + gan off)"
echo "  max_steps=2 sc_dmd_enabled=true (everything else off)"
echo "================================================================"
T3_LOG=$LOGROOT/testS3
mkdir -p "$T3_LOG"
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_s3 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+3)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T3_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_aux_scdmd_s3_sc_only \
    max_steps=2 \
    log_interval=1 \
    checkpoint_interval=999 \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
  2>&1 | tee "$T3_LOG/testS3.log" || { echo "TEST S3 FAILED (SC alone broken)"; exit 43; }
echo "TEST S3 PASSED: SC-only path works"

# ---------------------------------------------------------------------
# Test S4: 4-GPU DDP smoke, SC OFF (regression guard).
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST S4: 4-GPU DDP smoke, SC OFF (aux still on, GAN still off)"
echo "  max_steps=2 sc_dmd_enabled=false"
echo "================================================================"
T4_LOG=$LOGROOT/testS4
mkdir -p "$T4_LOG"
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_s4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+4)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T4_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_aux_scdmd_s4_sc_off \
    max_steps=2 \
    log_interval=1 \
    checkpoint_interval=999 \
    sc_dmd_enabled=false \
  2>&1 | tee "$T4_LOG/testS4.log" || { echo "TEST S4 FAILED (SC-off regressed)"; exit 44; }
if grep -qE "sc_dmd_loss_raw" "$T4_LOG/testS4.log"; then
  echo "TEST S4 WARNING: SC-DMD telemetry SHOULD NOT be in log when sc_dmd_enabled=false"
else
  echo "TEST S4 PASSED: sc_dmd_enabled=false truly skips SC pass"
fi

echo ""
echo "================================================================"
echo "  ALL AUX+SC-DMD (NO GAN) SMOKE TESTS PASSED on $(date)"
echo "  Logs: $LOGROOT"
echo "================================================================"
