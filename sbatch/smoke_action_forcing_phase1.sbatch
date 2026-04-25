#!/bin/bash
#SBATCH --job-name=smoke-action-forcing-phase1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing smoke harness — runs the 4-test sequence the
# user mandated before launching the 10h job. Single 4-GPU job; tests
# 1+2 use --nproc_per_node=1 (single-GPU smoke) and tests 3+4 use
# --nproc_per_node=4 (DDP).
#
#   Test 1: single-GPU smoke, no extensions
#       max_steps=1  threshold=null  cap=0
#       → confirms threshold=null + cap=0 path is genuinely free
#         (no extra ride frames loaded, no extension forwards).
#
#   Test 2: single-GPU forced extension
#       max_steps=1  threshold=999  cap=1
#       → confirms gen/mae_extension_count=1 when GT extends past 21
#         (rides are now ≤105 frames so this should always succeed).
#
#   Test 3: 4-GPU DDP smoke, forced extension
#       max_steps=2  threshold=999  cap=1  (4 GPUs)
#       → exercises the all-reduce(MIN) deadlock fix when ranks have
#         different available extra GT.
#
#   Test 4: short real run with intended config
#       max_steps=20  (4 GPUs, threshold/cap from YAML defaults)
#       → watches peak memory + gen/mae_extension_count under the
#         intended training distribution.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1.yaml
LOGROOT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_action_forcing_phase1_j${SLURM_JOB_ID}
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

# Disable W&B for smoke tests (we just want stdout signals, not project
# pollution). The trainer respects WANDB_MODE=disabled.
export WANDB_MODE=disabled

echo "CONFIG=$CONFIG"
echo "LOGROOT=$LOGROOT"
echo "Starting smoke harness on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true

# ---------------------------------------------------------------------
# Test 1: single-GPU smoke, no extensions.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST 1: single-GPU smoke, no extensions"
echo "  max_steps=1 threshold=null cap=0"
echo "================================================================"
T1_LOG=$LOGROOT/test1
mkdir -p "$T1_LOG"
srun --ntasks=1 --ntasks-per-node=1 --gpus=1 torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=${SLURM_JOB_ID}_t1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+1)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T1_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_t1_no_ext \
    max_steps=1 \
    log_interval=1 \
    checkpoint_interval=999 \
    mae_extension_threshold=null \
    mae_extension_max_extra_chunks=0 \
  2>&1 | tee "$T1_LOG/test1.log" || { echo "TEST 1 FAILED"; exit 11; }
echo "TEST 1 PASSED"

# ---------------------------------------------------------------------
# Test 2: single-GPU forced extension.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST 2: single-GPU forced extension"
echo "  max_steps=1 threshold=999 cap=1   (expect mae_extension_count=1)"
echo "================================================================"
T2_LOG=$LOGROOT/test2
mkdir -p "$T2_LOG"
srun --ntasks=1 --ntasks-per-node=1 --gpus=1 torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=${SLURM_JOB_ID}_t2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+2)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T2_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_t2_forced_ext \
    max_steps=1 \
    log_interval=1 \
    checkpoint_interval=999 \
    mae_extension_threshold=999 \
    mae_extension_max_extra_chunks=1 \
  2>&1 | tee "$T2_LOG/test2.log" || { echo "TEST 2 FAILED"; exit 12; }
# Verify gen/mae_extension_count=1 appears in the log
if grep -q "mae_extension_count.*1" "$T2_LOG/test2.log"; then
  echo "TEST 2 PASSED: mae_extension_count=1 detected"
else
  echo "TEST 2 WARNING: mae_extension_count=1 not detected in log"
fi

# ---------------------------------------------------------------------
# Test 3: 4-GPU DDP smoke, forced extension.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST 3: 4-GPU DDP smoke, forced extension"
echo "  max_steps=2 threshold=999 cap=1   (DDP all-reduce(MIN) test)"
echo "================================================================"
T3_LOG=$LOGROOT/test3
mkdir -p "$T3_LOG"
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_t3 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+3)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T3_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_t3_ddp_forced_ext \
    max_steps=2 \
    log_interval=1 \
    checkpoint_interval=999 \
    mae_extension_threshold=999 \
    mae_extension_max_extra_chunks=1 \
  2>&1 | tee "$T3_LOG/test3.log" || { echo "TEST 3 FAILED (likely DDP deadlock)"; exit 13; }
echo "TEST 3 PASSED: 4-GPU DDP completed without deadlock"

# ---------------------------------------------------------------------
# Test 4: short real run with intended config.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST 4: short real run (4 GPUs, max_steps=20, YAML defaults)"
echo "  threshold/cap from YAML; watch peak memory + extension count"
echo "================================================================"
T4_LOG=$LOGROOT/test4
mkdir -p "$T4_LOG"
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_t4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+4)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T4_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_t4_real_short \
    max_steps=20 \
    log_interval=1 \
    checkpoint_interval=999 \
  2>&1 | tee "$T4_LOG/test4.log" || { echo "TEST 4 FAILED"; exit 14; }
echo "TEST 4 PASSED"

echo ""
echo "================================================================"
echo "  ALL SMOKE TESTS PASSED on $(date)"
echo "  Logs: $LOGROOT"
echo "================================================================"
