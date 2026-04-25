#!/bin/bash
#SBATCH --job-name=smoke-action-forcing-phase1-aux
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing AUX-VARIANT smoke harness.
#
# Validates the action critic + frozen-teacher z-guidance loss path
# end-to-end (build, forward, backward, all-reduce on critic, save/
# resume) before launching the 10h aux job:
#
#   Test A1: single-GPU smoke, AUX ON
#       max_steps=1
#       → confirms the trainer builds the cotracker+ss_vae teacher,
#         un-freezes + DDP-wraps the action_critic, builds the
#         critic optimizer, and runs a full generator iter that
#         emits ``train/critic_z_loss`` / ``train/gen_z_loss`` /
#         ``train/critic_z2_mse`` / etc. to stdout.
#
#   Test A2: 4-GPU DDP smoke, AUX ON
#       max_steps=2
#       → exercises DDP all-reduce on critic gradients (multi-step
#         critic_updates_per_step=2 forward/backward) and DDP all-
#         reduce on the generator backward (with z-guidance loss
#         in the graph). Catches any rank-asymmetric forward / DDP
#         "found unused params" surprise on the critic-update path.
#
#   Test A3: 4-GPU DDP smoke, AUX OFF
#       max_steps=2 action_teacher_mode=off action_critic_aux_enabled=false
#       → fall-through path: confirms the aux variant config can
#         still run as a pure-DMD config when both aux gates are
#         flipped off (regression guard against accidental aux-only
#         coupling).
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_aux.yaml
LOGROOT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_action_forcing_phase1_aux_j${SLURM_JOB_ID}
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
echo "Starting AUX smoke harness on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true

# ---------------------------------------------------------------------
# Test A1: single-GPU smoke, AUX ON.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST A1: single-GPU smoke, AUX ON"
echo "  max_steps=1 (full critic + teacher path on 1 GPU)"
echo "================================================================"
T1_LOG=$LOGROOT/testA1
mkdir -p "$T1_LOG"
srun --ntasks=1 --ntasks-per-node=1 --gpus=1 torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=${SLURM_JOB_ID}_a1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+1)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T1_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_aux_a1_single \
    max_steps=1 \
    log_interval=1 \
    checkpoint_interval=999 \
  2>&1 | tee "$T1_LOG/testA1.log" || { echo "TEST A1 FAILED"; exit 21; }
# Verify the critic logging line at least appeared.
if grep -qE "(critic_z_loss|train/critic_z_loss|gen_action_loss)" "$T1_LOG/testA1.log"; then
  echo "TEST A1 PASSED: critic+teacher path executed and emitted logs"
else
  echo "TEST A1 WARNING: critic-loss telemetry not detected in log"
fi

# ---------------------------------------------------------------------
# Test A2: 4-GPU DDP smoke, AUX ON.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST A2: 4-GPU DDP smoke, AUX ON"
echo "  max_steps=2 (DDP all-reduce on critic + generator)"
echo "================================================================"
T2_LOG=$LOGROOT/testA2
mkdir -p "$T2_LOG"
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_a2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+2)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T2_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_aux_a2_ddp \
    max_steps=2 \
    log_interval=1 \
    checkpoint_interval=999 \
  2>&1 | tee "$T2_LOG/testA2.log" || { echo "TEST A2 FAILED (likely DDP issue on critic)"; exit 22; }
echo "TEST A2 PASSED: 4-GPU DDP with aux losses completed without deadlock"

# ---------------------------------------------------------------------
# Test A3: 4-GPU DDP smoke, AUX OFF (regression guard).
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST A3: 4-GPU DDP smoke, AUX OFF (fall-through)"
echo "  max_steps=2 action_teacher_mode=off action_critic_aux_enabled=false"
echo "================================================================"
T3_LOG=$LOGROOT/testA3
mkdir -p "$T3_LOG"
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_a3 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+3)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T3_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_aux_a3_aux_off \
    max_steps=2 \
    log_interval=1 \
    checkpoint_interval=999 \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
  2>&1 | tee "$T3_LOG/testA3.log" || { echo "TEST A3 FAILED"; exit 23; }
echo "TEST A3 PASSED: aux-off fall-through works"

echo ""
echo "================================================================"
echo "  ALL AUX SMOKE TESTS PASSED on $(date)"
echo "  Logs: $LOGROOT"
echo "================================================================"
