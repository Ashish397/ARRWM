#!/bin/bash
#SBATCH --job-name=smoke-extender
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:45:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# SMOKE: extender + eval-video sanity check (STREAMING path).
#
# Runs the action-forcing trainer in STREAMING mode for 6 iters per
# phase, on hand-picked low-motion / high-motion ride pools. Tests the
# new streaming extender wired into trainer._streaming_maybe_extend.
#
# Expected behaviour after both phases finish:
#
#   * gen/streaming_extension_count on the LOW phase wandb run should
#     be HIGH (the extender keeps pushing the streaming sequence
#     forward by no-grad chunks because per-chunk MAE stays below 0.5
#     — model is good at predicting "nothing happens").
#
#   * gen/streaming_extension_count on the HIGH phase wandb run should
#     be LOW (per-chunk MAE crosses 0.5 quickly because the scene is
#     moving hard enough that the student's prediction can't keep up).
#
#   * Both runs upload sample_interval=2 videos to wandb under
#     ``sample/pred_image`` AND save them locally to
#     <log_dir>/samples/step_NNN.mp4. Steps with videos: 1, 2, 4, 6.
#
# After the job completes, run::
#
#   python tests/check_smoke_extender.py
#
# to pull the two runs from wandb and assert ext_count_low > ext_count_high.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/smoke_extender.yaml
LOGROOT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_extender
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR" logs "$LOGROOT"

MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))

export NCCL_CROSS_NIC=1
export NCCL_SOCKET_IFNAME=hsn
export NCCL_DEBUG=WARN
export NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---------------------------------------------------------------------
# Phase A — LOW motion. Extender SHOULD extend repeatedly.
# ---------------------------------------------------------------------
echo "================================================================="
echo "[$(date)] PHASE A: LOW-MOTION smoke"
echo "================================================================="
PHASE_LOG=$LOGROOT/phase_low
mkdir -p "$PHASE_LOG"
srun torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_smoke_low \
    log_dir=$PHASE_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_extender_low_j${SLURM_JOB_ID}

echo "[$(date)] PHASE A complete. Local samples in: $PHASE_LOG/samples/"

# ---------------------------------------------------------------------
# Phase B — HIGH motion. Extender SHOULD cut off early.
# ---------------------------------------------------------------------
echo ""
echo "================================================================="
echo "[$(date)] PHASE B: HIGH-MOTION smoke"
echo "================================================================="
PHASE_LOG=$LOGROOT/phase_high
mkdir -p "$PHASE_LOG"

# Bump the rendezvous port so the second torchrun doesn't collide with
# any lingering rdzv state from the first.
MASTER_PORT=$((29501 + SLURM_JOB_ID % 16000))

srun torchrun \
  --nnodes=1 \
  --nproc_per_node=1 \
  --rdzv_id=${SLURM_JOB_ID}b \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_smoke_high \
    log_dir=$PHASE_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=smoke_extender_high_j${SLURM_JOB_ID}

echo "[$(date)] PHASE B complete. Local samples in: $PHASE_LOG/samples/"

echo ""
echo "================================================================="
echo "Smoke job done. Run the analysis to assert:"
echo "  python tests/check_smoke_extender.py --job_id ${SLURM_JOB_ID}"
echo "================================================================="
