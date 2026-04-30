#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-freeze-16h-v2
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
# Phase-1 Action-Forcing DMD — DEFAULTS + TEACHER-FREEZE MASK + ACTION
# AUX STACK, 1 node x 4 GPUs, 16h.
#
# Same as ``train_action_forcing_phase1_16h.sbatch`` (locked-in DMD
# fundamentals: +npb anchor, last-chunk DMD mask, full-bidir scorer,
# streaming on, collapse gate avg-rollout-MAE > 0.5) PLUS:
#
#   1. Per-slot / per-frame teacher-freeze rejection on the DMD gradient.
#   2. The full action-aux stack live so the student keeps learning the
#      correct action even when DMD is masked off on a slot:
#        * action_teacher_mode = "all"   (CoTracker + ss_vae loaded)
#        * action_critic_aux_enabled = true   (z-guidance gradient)
#        * state_probe_aux_enabled = true   (state probe head)
#        * generator state-token branch active (from v14 LoRA)
#
# Why teacher-freeze masking. The v14 real_score (frozen LoRA-merged
# bidirectional teacher) occasionally outputs a single hazy / frozen
# latent frame inside the 21-frame DMD scoring window — visible as a
# washed-out frame in ``_diag_p1/out/F_real_GT.mp4`` around the 3-sec
# mark. If we let DMD score against that frame the (fake - real)
# gradient pulls the student toward the bad teacher pred and the
# student learns to reproduce the freeze.
#
# Two detection modes, selectable via ``teacher_freeze_mode``:
#
#   * "mae" (legacy): per-frame MAE between real_score's pred and the
#     GT video at the same time positions; flag frames where MAE >
#     ``teacher_freeze_threshold`` × median(MAE). Calibration sweep
#     (_diag_p1/freeze_threshold_calibration.py over 16 motion-rich
#     windows) shows natural max-vs-median ratios cap at ~1.4× in the
#     middle and ~1.8× at the structural boundary, so the default
#     ``threshold=2.0`` only fires on severe freezes.
#
#   * "action" (NEW): run BOTH the real_score's pred AND the GT video
#     through the frozen action teacher (CoTracker → ss_vae → 8-d z)
#     and flag SLOTS where cosine_similarity(z_real, z_GT) <
#     ``teacher_freeze_action_threshold``. Default 0.0 = "teacher
#     direction OPPOSITE to GT direction" — a teacher that gets the
#     action MAGNITUDE wrong but the DIRECTION right stays unmasked
#     (cos sim is magnitude-invariant). We compare to GT (not to
#     student) so we don't mask slots where the teacher is correct
#     and the student is the one drifting — those slots are exactly
#     what DMD should keep training on.
#     Diagnostic verification: ``_diag_p1/test_freeze_action_mode.py``
#     on ride 20240131104731.zarr (which contains a backwards-driving
#     segment) at t=178 flagged 8/21 slots with cos(R,GT) negative —
#     all on the strong-backwards commands (z7 ≈ -0.9). cos(R,S) was
#     +0.99 on those slots = teacher and student both got direction
#     wrong but agreed; the cos(R,GT) metric correctly catches that.
#     The action critic + state probe aux losses keep the student
#     learning the correct action through the masked slots.
#     ``ActionForcingDMD.__init__`` asserts the action-aux stack is
#     wired before this mode can be used.
#
# Knobs (configs/action_forcing_phase1_freeze.yaml):
#   * teacher_freeze_detect_enabled: true
#   * teacher_freeze_mode: "mae" | "action"
#   * teacher_freeze_threshold: 2.0   (used in "mae" mode)
#   * teacher_freeze_action_threshold: 0.0   (used in "action" mode;
#                                              cosine_similarity(z_real, z_GT)
#                                              cutoff)
#
# Per-iter wandb metrics (gen/ namespace):
#   * Always:    gen/teacher_freeze_count, gen/teacher_freeze_rate
#   * mode=mae:  gen/teacher_freeze_max_mae, gen/teacher_freeze_median_mae
#   * mode=action: gen/teacher_freeze_action_cos_RG_min/_mean/_median
#                                              (decision metric, vs GT)
#                  gen/teacher_freeze_action_cos_RS_mean
#                                              (context only, vs student)
#
# Sister sbatches:
#   sbatch/train_action_forcing_phase1_16h.sbatch        (defaults only)
#   sbatch/train_action_forcing_phase1_gan_16h.sbatch    (defaults + R3GAN)
#   sbatch/train_action_forcing_phase1_scdmd_16h.sbatch  (defaults + SC-DMD)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_freeze.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_freeze_16h_v2
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export HUGGINGFACE_HUB_CACHE=$CACHE_DIR
export TRANSFORMERS_CACHE=$CACHE_DIR
mkdir -p "$CACHE_DIR"

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
echo "Starting Phase-1 Action-Forcing DMD + teacher-freeze mask (1 node x 4 GPUs, 16h) on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

srun torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=$SLURM_JOB_ID \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$LOGDIR \
    wandb_dir=$WANDB_SAVE_DIR \
    real_guidance_scale=0.0 \
    action_teacher_mode=all \
    max_rolls_per_ride=300 \
    dfake_gen_update_ratio=1 \
    streaming_max_length=1500 \
    max_ride_frames=1490 \
    run_name=action_forcing_phase1_freeze_j${SLURM_JOB_ID}_v2

echo "Training completed on $(date)"
