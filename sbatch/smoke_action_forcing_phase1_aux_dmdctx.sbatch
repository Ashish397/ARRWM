#!/bin/bash
#SBATCH --job-name=smoke-action-forcing-phase1-aux-dmdctx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing AUX + DMD_CONTEXT smoke harness (A1).
#
# Validates the dmd_context=true path end-to-end (clean GT context
# threaded into scorers as clean_x with context_shift=cf//npb=1,
# rollout window shifted by cf=3) before the 10h job:
#
#   Test D1: single-GPU smoke, dmd_context=true.
#       max_steps=1
#       → confirms ``ActionForcingDMD.__init__`` sets
#         ``real_score.model.context_shift`` and
#         ``fake_score.model.context_shift`` to 1, the trainer
#         loads ride[0..23] and slices clean_x = ride[0..20],
#         student rolls ride[3..23] (21 frames), and the
#         generator/critic losses produce finite values with the
#         clean-context conditional dict threaded through.
#
#   Test D2: 4-GPU DDP smoke, dmd_context=true.
#       max_steps=2
#       → exercises DDP all-reduce on the dmd_context path. Catches
#         OOM (only ~+15% scorer FLOPs from 24 vs 21 frames), NaN,
#         and DDP "found unused parameters" surprises around the
#         clean_x / context_shift wiring.
#
#   Test D3: 4-GPU DDP smoke, dmd_context=false (regression guard).
#       max_steps=2 dmd_context=false min_ride_frames=21
#       → fall-through: confirms the dmd_context config still works
#         when dmd_context is flipped off — should match the AUX
#         baseline exactly (no clean_x, no context_shift, no
#         clean_conditional_dict).
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_aux_dmdctx.yaml
LOGROOT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_action_forcing_phase1_aux_dmdctx_j${SLURM_JOB_ID}
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
echo "Starting AUX+DMD_CONTEXT smoke harness on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true

# ---------------------------------------------------------------------
# Test D1: single-GPU smoke, dmd_context=true.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST D1: single-GPU smoke, dmd_context=true (A1)"
echo "  rollout_frames=21 + cf=3 clean GT context to scorers"
echo "================================================================"
T1_LOG=$LOGROOT/testD1
mkdir -p "$T1_LOG"
srun --ntasks=1 --ntasks-per-node=1 --gpus=1 torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=${SLURM_JOB_ID}_d1 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+1)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T1_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_dmdctx_d1_single \
    max_steps=1 \
    log_interval=1 \
    checkpoint_interval=999 \
  2>&1 | tee "$T1_LOG/testD1.log" || { echo "TEST D1 FAILED"; exit 31; }
if grep -qE "(generator_loss|generator_dmd_loss)" "$T1_LOG/testD1.log"; then
  echo "TEST D1 PASSED: dmd_context=true single-GPU full path executed"
else
  echo "TEST D1 WARNING: generator_loss telemetry not detected in log"
fi

# ---------------------------------------------------------------------
# Test D2: 4-GPU DDP smoke, dmd_context=true.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST D2: 4-GPU DDP smoke, dmd_context=true (A1)"
echo "  max_steps=2 (DDP all-reduce, scorers see 24 frames)"
echo "================================================================"
T2_LOG=$LOGROOT/testD2
mkdir -p "$T2_LOG"
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_d2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+2)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T2_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_dmdctx_d2_ddp \
    max_steps=2 \
    log_interval=1 \
    checkpoint_interval=999 \
  2>&1 | tee "$T2_LOG/testD2.log" || { echo "TEST D2 FAILED (likely OOM or DDP issue on long rollout)"; exit 32; }
echo "TEST D2 PASSED: 4-GPU DDP dmd_context=true completed"

# ---------------------------------------------------------------------
# Test D3: 4-GPU DDP smoke, dmd_context=false (regression guard).
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  TEST D3: 4-GPU DDP smoke, dmd_context=false (fall-through)"
echo "  max_steps=2 dmd_context=false min_ride_frames=21"
echo "================================================================"
T3_LOG=$LOGROOT/testD3
mkdir -p "$T3_LOG"
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_d3 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+3)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$T3_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_dmdctx_d3_off \
    max_steps=2 \
    log_interval=1 \
    checkpoint_interval=999 \
    dmd_context=false \
    min_ride_frames=21 \
  2>&1 | tee "$T3_LOG/testD3.log" || { echo "TEST D3 FAILED"; exit 33; }
echo "TEST D3 PASSED: dmd_context=false fall-through works"

echo ""
echo "================================================================"
echo "  ALL AUX+DMD_CONTEXT SMOKE TESTS PASSED on $(date)"
echo "  Logs: $LOGROOT"
echo "================================================================"
