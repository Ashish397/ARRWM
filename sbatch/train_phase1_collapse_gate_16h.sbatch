#!/bin/bash
#SBATCH --job-name=phase1-collapse-gate-16h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Rolling-Staircase DMD, 1 node x 4 GPUs, 16h.
#
# COLLAPSE-GATE + CURRICULUM variant. The student gets adaptively-
# truncated rollouts: every rolling step the batcher computes a
# per-chunk MAE between the just-committed slot-0 chunk and its
# GT-aligned counterpart -- ``(stu_c - gt_c).abs().mean()``. As soon
# as that exceeds ``collapse_mae_threshold`` (= 0.65 in the YAML),
# the slot is flagged for refill at the NEXT prepare_for_iter
# boundary. The current 8-step rolling cycle still completes on
# every rank (DDP lockstep preserved -- single allreduce on the
# final backward), then the flagged slot pulls a fresh ride.
# Combined with ``sort_rides_by_length: asc`` we get a curriculum
# that starts on short stable rides and only progresses to longer
# rides once the student isn't tripping the gate.
#
# All knobs come from the YAML; this sbatch only sets:
#   * log_dir          -> per-job log + checkpoint directory
#   * wandb_dir        -> wandb scratch space (gitignored)
#   * run_name         -> wandb run name (distinct from the 2x2 sbatch)
#
# YAML invariants (do NOT override here -- if you want to change
# them, edit configs/longlive_phase1_rolling_staircase.yaml):
#   * num_live_slots          = 4   (4-slot staircase)
#   * passes_per_step         = 1
#   * denoising_step_list     = [1000, 625, 500, 312.5]   (ODE-aligned)
#   * slots_per_rank          = 4
#   * rolling_steps_per_iter  = 8
#   * dmd_active_slot_policy  = all (Option 4 batched-S_n DMD)
#   * action_teacher_mode     = slot0 (CoTracker + ss_vae teacher on)
#   * real_score_num_gt_chunks = 2
#   * sort_rides_by_length    = asc (shortest -> longest)
#   * collapse_mae_threshold  = 0.65
#   * min_ride_frames         = 160
#
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/longlive_phase1_rolling_staircase.yaml
# NOTE on the actual checkpoint directory:
#   trainer.causal_rolling_staircase_train.log_dir appends ``run_name``
#   (= ``phase1_rolling_staircase`` from the YAML) to whatever ``log_dir``
#   we pass in here. So checkpoints, vis, and the auto-resume scan land
#   under ``${LOGDIR}/phase1_rolling_staircase/`` -- NOT directly in
#   ``${LOGDIR}``. Set up to match because some tooling globs there.
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/phase1_collapse_gate_16h
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

# action_teacher_mode=slot0 → CoTracker + ss_vae are loaded. Leave
# TORCH_HOME on its default (~/.cache/torch) so CoTracker3's offline
# repo cache is reachable; ditto cotracker_source_dir /
# cotracker_checkpoint_path in the YAML.

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
echo "Starting phase-1 collapse-gate + curriculum DMD (1 node, 4 GPUs, 16h) on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Minimal overrides: paths + wandb run name. Everything else experiment-
# shaped stays in the YAML. run_name is phase1_4x1_* vs train_phase1_
# collapse_gate_2x2_16h.sbatch (phase1_2x2_*).
srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_rolling_staircase_train.py \
  --config $CONFIG \
  --override \
    log_dir=$LOGDIR \
    wandb_dir=$WANDB_SAVE_DIR \
    run_name=phase1_collapse_4x1_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
