#!/bin/bash
#SBATCH --job-name=phase1-rolling-staircase-1node-5h-noaux
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=5:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Rolling-Staircase DMD, single-node (4 GPU), 5h run.
#
# NO-AUX ABLATION: identical to the main 5h sbatch in every dimension
# except the action-supervision apparatus, which is stripped down to
# "DMD + AdaLN/Stream-A/Stream-B only":
#
#   DISABLED via --override:
#     * action_critic_aux_enabled=false
#         → no critic instantiation, no critic aux loss.
#     * state_probe_aux_enabled=false
#         → no state-probe module attached to the generator wrapper
#           (also bypasses the probe forward in wan_wrapper, so no
#           tap-activation overhead inside the DiT).
#     * action_teacher_mode=off
#         → no CoTracker / ss_vae forward; no teacher-z targets.
#           The remaining aux paths (if any) would fall back to their
#           commanded-action branches, but since both aux heads are off
#           the teacher is pure dead code in this run.
#
#   KEPT ON AND TRAINABLE:
#     * action_projection (Stream A → AdaLN modulation): on + trainable.
#         Same `train_action_projection: true` as the main config.
#     * action_token_projection (Stream B → per-frame action tokens):
#         on + trainable. Same `train_action_token_projection: true`.
#     * DiT generator + fake_score + real_score trained as in main config.
#
# Purpose of this variant: isolate the DMD-only learning signal through
# the two trainable action-projection MLPs. If the main run regresses
# or over-fits to the state_probe/action_critic heads, this ablation
# tells us what the student learns purely from the teacher/fake-score
# DMD loss + the per-frame action stream, without any "hint" from the
# motion-pipeline action teacher.
#
# Everything else — dataset, KV cache geometry, viz cadence, ride
# sorting, step-0 video, checkpoint loading — comes from the YAML as
# in the main 5h sbatch.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/longlive_phase1_rolling_staircase.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/phase1_rolling_staircase_1node_5h_noaux
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

# No CoTracker is loaded in this run (action_teacher_mode=off), so
# no ~/.cache/torch access is needed. TORCH_HOME left untouched.

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
echo "Starting phase-1 rolling-staircase DMD NO-AUX (1 node, 4 GPUs, 5h) on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Config overrides baked into the launch:
#   - log_dir                      -> per-job log dir (ckpt + local vis fallback)
#   - wandb_dir                    -> wandb scratch space (./wandb is gitignored)
#   - action_critic_aux_enabled    -> OFF
#   - state_probe_aux_enabled      -> OFF
#   - action_teacher_mode          -> "off" (string, to survive YAML 1.1
#     bool coercion; the dotlist parser would otherwise turn the bareword
#     `off` into Python's `False` — see utils/action_teacher.py).
#   - (train_action_projection / train_action_token_projection kept ON
#     via the YAML defaults; explicitly re-affirmed here to make the
#     ablation's intent obvious at launch time.)
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
    slots_per_rank=4 \
    rolling_steps_per_iter=16 \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    action_teacher_mode=off \
    train_action_projection=true \
    train_action_token_projection=true

echo "Training completed on $(date)"
