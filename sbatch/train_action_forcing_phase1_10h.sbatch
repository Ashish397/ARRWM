#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-10h
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=10:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing DMD (NO STAIRCASE), 1 node x 4 GPUs, 10h.
#
# CF-style self-forcing on a fixed 21-frame window with action
# awareness — byte-for-byte CF chunkwise geometry: 7 chunks of 3
# frames = 21 frames; chunks_per_rolling_step=1 ⇒ 7 rolling steps of
# 3 frames each (one chunk denoised per rolling step). Per training
# iter:
#   1. Pull a 21-frame ride.
#   2. Roll the student through 7 rolling steps (cps=1, sizes
#      [3,3,3,3,3,3,3]) with truncated random-exit denoising and a
#      persistent KV cache.
#   3. Compute CF-parity DMD loss against the v14 LoRA-merged
#      real_score and the generator-mirrored fake_score (real and
#      fake symmetric on the student's noisy 21-frame pred).
#   4. Generator backward + EMA update.
#   5. Compute CF-parity critic loss (fake_score denoise MSE on
#      the student's pred).
#   6. fake_score backward.
#
# All knobs come from configs/action_forcing_phase1.yaml; this sbatch
# only sets:
#   * log_dir  -> per-job log + checkpoint directory
#   * wandb_dir -> wandb scratch space (gitignored)
#   * run_name -> wandb run name (distinct from staircase sbatches)
#
# CF chunkwise DMD parity (see configs/action_forcing_phase1.yaml header):
#   * Generator AdamW: lr=2e-6, betas=(0.0, 0.999), wd=0.01
#   * Critic    AdamW: lr=4e-7, betas=(0.0, 0.999), wd=0.01
#   * ts_schedule=false, ema_weight=0.99, ema_start_step=200
#   * sink_size=3 (LongLive-style 3-frame anchor at the head of the
#     KV cache; no-op during training because cache doesn't roll, but
#     enables sink-aware AR rollout without retraining).
#   * KEPT-AS-UPGRADE: dfake_gen_update_ratio=1 (DMD2; CF DMD uses 5),
#     fake_score_updates_enabled=true.
#
# Infinity-RoPE: the YAML defaults to ``infinity_rope: true``, so the
# Action-Forcing trainer
# (trainer/causal_action_forcing_train.py::ActionForcingDMDTrainer
# ._build_pipeline) installs ``utils/infinity_rope.install()`` on
# the generator's DiT before the training loop starts. The student
# is therefore trained with the SAME RoPE convention that
# ``utils/eval_causal_AR.py`` uses at long-horizon AR rollout time
# (also defaulting to infinity_rope=true), preventing the
# train/inference mismatch that produced the visible chunk-boundary
# stutter on the original AR path. Within a CLASSIC
# (rollout_frames=21) Action-Forcing training iter the cache never
# rolls (num_training_frames = local_attn_size = 21; sink_size=3 is
# a no-op at this geometry), so the patch is mathematically
# equivalent to the original RoPE path during training. To run an
# A/B against the original RoPE path, append ``infinity_rope=false``
# to the --override line below.
#
# Long-rollout mode (CF parity, off by default): set ``rollout_frames``
# in the YAML or via --override (e.g. ``rollout_frames=42``) to roll
# the student through more frames than the gradient window. The
# leading ``rollout_frames - num_training_frames`` frames warm the
# KV cache under no_grad, and only the trailing ``num_training_frames``
# carry gradient — byte-for-byte CF's ``start_gradient_frame_index``
# at ``Causal-Forcing/pipeline/self_forcing_training.py:120``. When
# this is on, ``sink_size=3`` and ``local_attn_size=21`` become LIVE
# (the gradient-active forwards exercise the LongLive-style sink
# window). The first ``num_frame_per_block`` (=3) frames of the
# scoring window are also masked off the DMD loss as a CF-style
# BOUNDARY (their attention context is no_grad warmup-cache content,
# so penalising the student on them would just teach it to smooth
# the warmup→grad transition); CF reference at
# ``Causal-Forcing/long_video/model/base.py:169-177``. We deliberately
# SKIP CF's decode→re-encode of the boundary latent (a full VAE
# round-trip per training step) — with the gradient masked off,
# the boundary frame's contents cannot affect the loss anyway.
# Bump ``min_ride_frames`` and ``max_ride_frames`` in the YAML (or
# via --override) in lockstep so the loader can find rides long
# enough.
#
# MAE-driven dynamic rollout extension (ON by default in the YAML —
# threshold=0.5, max_extra_chunks=13). After the baseline 21-frame
# rollout, the pipeline checks the latent-MAE between the last 3
# generated frames and GT. If MAE < threshold AND ride still has GT
# AND we're under the cap, it rolls one more 3-frame chunk under
# no_grad (full denoise, KV-cache-committed) and re-checks. Loops
# until: MAE crosses threshold, ride GT exhausted, or cap hit.
# Extensions are NEVER scored — DMD loss still operates on the
# baseline last ``num_training_frames``. Per-iter wandb metrics:
#   gen/baseline_last_chunk_mae   (always logged when GT is passed)
#   gen/last_chunk_mae            (final-chunk MAE — same as
#                                  baseline_last_chunk_mae when no
#                                  extensions fired)
#   gen/mae_extension_count       (extensions rolled this iter)
# Memory cost: ZERO extra over baseline. Extensions commit through
# the model's rolling KV cache (sink-pinned + FIFO local window), so
# the buffer stays at ``max(num_max_frames, rollout_frames)`` frames
# regardless of the cap. ``mae_extension_max_extra_chunks`` bounds
# COMPUTE only (each extension is an extra full-denoise + cache-
# commit forward under no_grad).
# Disable the loop with ``--override mae_extension_threshold=null``
# (still keeps the baseline MAE metric on) or
# ``--override mae_extension_max_extra_chunks=0`` (same effect,
# threshold reading preserved).
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1.yaml
# NOTE on the actual checkpoint directory:
#   trainer.causal_rolling_staircase_train.log_dir (inherited by
#   trainer.causal_action_forcing_train) appends ``run_name``
#   (= ``action_forcing_phase1`` from the YAML) to whatever
#   ``log_dir`` we pass in here. So checkpoints, vis, and the auto-
#   resume scan land under ``${LOGDIR}/action_forcing_phase1/`` --
#   NOT directly in ``${LOGDIR}``. Set up to match because some
#   tooling globs there.
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_10h
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
echo "Starting phase-1 Action-Forcing DMD (1 node, 4 GPUs, 10h) on $(date)"

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
    run_name=action_forcing_phase1_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
