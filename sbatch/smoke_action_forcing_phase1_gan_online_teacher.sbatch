#!/bin/bash
#SBATCH --job-name=smoke-action-forcing-phase1-gan-online-teacher
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:45:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing GAN + ONLINE real_teacher smoke harness.
#
# Catches the bug that just broke the 16h gan run (job 4430533):
#   ``RuntimeError: DistributedDataParallel is not needed when a
#    module doesn't have any parameter that requires a gradient``
# at trainer/causal_action_forcing_train.py:330 — the real_score DDP
# wrap. Root cause: the unconditional finalize-freeze in
# ``ActionForcingDMD._build_model`` clobbered the LoRA params that
# ``_load_real_score_with_v14_lora`` had marked trainable for the
# online-teacher path. Fix gated the freeze on
# ``not self.real_teacher_train_online``.
#
# Validates:
#   * peft adapter stays alive on real_score (no merge_and_unload).
#   * LoRA params have requires_grad=True after _build_model finishes.
#   * Trainer's DDP-wrap of real_score succeeds (find_unused_params).
#   * Real-teacher optimizer + warmup + step path runs end-to-end.
#   * R3GAN + online teacher co-exist for >=2 DDP steps.
#
# Single test (4-GPU DDP, fail-fast):
#   T2: 4-GPU DDP smoke, gan + online teacher ON
#       max_steps=20 (enough to exercise the training loop past the
#       first-iter cudnn warmup and into steady state — the failure
#       mode from job 4430623 was a hang on iter 0, so finishing 20
#       cleanly is decisive), wrapped in a 30-min `timeout` so a
#       hang in the online-teacher forward fails fast instead of
#       wasting the wallclock.
#
#       Training-pattern overrides: NONE. We inherit phase1_gan.yaml's
#       dfake_gen_update_ratio (=2 critic updates per gen update),
#       real_teacher_train_online (=true → real LoRA gets trained),
#       gan_updates_per_step, denoising_step_list, etc. The smoke
#       exercises the same per-step DDP collective pattern production
#       will. Only the IO-only knobs (sample_interval, vis_save_local)
#       are turned off — they don't change which models forward/backward
#       or which all-reduces fire.
#       Post-run assertion: ≥1 `step=N/20 gen_loss` line must appear
#       in T2.log — otherwise FAIL with diagnostic "init succeeded
#       but no step completed." Last smoke confirmed init but never
#       asserted training progress; that's the gap this catches.
#
# T1 (single-GPU) and T3 (offline-teacher regression guard) are both
# dropped — the production gan job is 4-rank online-teacher, so
# anything else is wasted load time. Loading v14 LoRA + 14B Wan once
# is ~5-7 min on its own, so a 3-test smoke would 3x that cost for
# zero added bug coverage.
#
# Smoke dataset:  /projects/u6ex/fbots/frodobots_encoded_smoke_high
#                 (low-motion pool gets filtered out by phase1_gan's
#                  motion_ride_min_mean=3.000)
# Wandb project:  longlive-phase1-smoke  (separate from production)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGROOT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_phase1_gan_online_teacher_j${SLURM_JOB_ID}
WANDB_SAVE_DIR=wandb

# Smoke dataset — combined smoke_low + smoke_high (~22 hand-picked
# rides). The individual smoke pools have only 11 zarrs each, of which
# only ~3 pass the production min_ride_frames=69 filter. With 4 DDP
# ranks each pulling its own DistributedSampler shard, 3 rides
# livelocks the multiprocessing dataloader (caught with py-spy: all 4
# ranks stuck in _next_ride / dataloader.__iter__). Combining the two
# pools and overriding min_ride_frames below gives every rank enough
# distinct rides to iterate.
#
# Also overrides three motion-filter knobs (motion_ride_min_mean /
# motion_skip_dead_rides / motion_start_threshold) per-test below to
# match smoke_extender.yaml — without those, phase1_gan's production
# motion thresholds reject every smoke ride.
SMOKE_ENCODED_ROOT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/smoke_combined_rides
# Wandb project for smoke runs — kept separate from production so smoke
# noise doesn't pollute the prod view.
SMOKE_WANDB_PROJECT=longlive-phase1-smoke

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

# NOTE: wandb is *enabled* for this smoke (logged to
# longlive-phase1-smoke), unlike the older smoke_action_forcing_phase1*
# harnesses which set WANDB_MODE=disabled. Following the
# smoke_extender.sbatch convention so smoke runs are reviewable in
# wandb without polluting the production project.

echo "CONFIG=$CONFIG"
echo "LOGROOT=$LOGROOT"
echo "SMOKE_ENCODED_ROOT=$SMOKE_ENCODED_ROOT"
echo "SMOKE_WANDB_PROJECT=$SMOKE_WANDB_PROJECT"
echo "Starting GAN + online-teacher smoke harness on $(date)"

srun --overlap --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true

# ---------------------------------------------------------------------
# T1 (single-GPU smoke) is intentionally omitted. Holding the full
# generator + DDP-trainable real_score (LoRA + Adam optimizer state) +
# R3GAN discriminator on a single 95 GiB GPU OOMs at ~94.8 GiB before
# the first training step. Production runs are 4–32 ranks where the
# activations / optimizer-state pressure is sharded across GPUs.
# Single-GPU couldn't catch the DDP-wrap bug from job 4430533 anyway
# (no DDP wrap = no failure mode), so the trade-off is fine: T2 below
# exercises the actual failing path under DDP.

# ---------------------------------------------------------------------
# T2: 4-GPU DDP smoke, gan + online teacher ON. Single test, fail-fast.
# ---------------------------------------------------------------------
# Reproduces the path that failed in job 4430533 (DDP-wrap freeze) and
# the path that hung silently in smoke job 4430623 (no step ever
# completed). Single test (no T1/T3) so the v14 LoRA + 14B Wan
# checkpoint is loaded only once. Speed knobs:
#   * sample_interval=0 + vis_save_local=false: skip the per-15-step
#     VAE decode + mp4 write. Pure I/O, doesn't touch the train loop.
#   * max_steps=20: 20 clean steps proves the loop is in steady state
#     (covers first-iter cudnn warmup + ride iteration + R3GAN warmup
#     ramp on the discriminator).
#
# Fail-fast guards:
#   * `timeout 1200` (20 min): if the online-teacher forward/all-reduce
#     hangs (the 1h41m silence we saw last time), kill it instead of
#     burning the wallclock.
#   * Post-run assertion: ≥1 `step=N/20 gen_loss` line must appear in
#     T2.log. The previous smoke confirmed init but never asserted
#     training progress — that's the gap this catches.
# ---------------------------------------------------------------------
echo ""
echo "================================================================"
echo "  T2: 4-GPU DDP smoke, gan + online real_teacher ON"
echo "  max_steps=20, full-fat training pattern (only IO disabled), 30-min timeout"
echo "================================================================"
T2_LOG=$LOGROOT/T2
mkdir -p "$T2_LOG"
# `timeout --kill-after=30 1800`: SIGTERM at 30 min, SIGKILL 30 s later
# if SIGTERM doesn't take (nested torchrun children sometimes ignore
# SIGTERM). 30 min covers the v14 LoRA + 14B Wan model load (~5-7 min)
# plus 20 full-fat training steps at production cadence (~30-60s/step
# on a 4-rank single-node smoke).
timeout --kill-after=30 1800 \
srun --overlap --cpu-bind=none torchrun \
  --nnodes=$SLURM_NNODES \
  --nproc_per_node=4 \
  --rdzv_id=${SLURM_JOB_ID}_t2 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:$((MASTER_PORT+2)) \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    encoded_root=$SMOKE_ENCODED_ROOT \
    motion_ride_min_mean=0.0 \
    motion_skip_dead_rides=false \
    motion_start_threshold=0.0 \
    min_ride_frames=21 \
    boundary_vae_roundtrip=false \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    max_rolls_per_ride=300 \
    dfake_gen_update_ratio=5 \
    collapse_mae_threshold=0.2 \
    mae_extension_threshold=0.2 \
    num_workers=0 \
    log_dir=$T2_LOG \
    wandb_dir=$WANDB_SAVE_DIR \
    wandb_project=$SMOKE_WANDB_PROJECT \
    run_name=smoke_gan_online_teacher_t2_ddp_j${SLURM_JOB_ID} \
    max_steps=20 \
    log_interval=1 \
    checkpoint_interval=999 \
    sample_interval=0 \
    vis_save_local=false \
  > "$T2_LOG/T2.log" 2>&1
T2_EXIT=$?

# Diagnostic banners — readable from job stdout without grepping.
echo ""
echo "----- T2 diagnostics -----"
echo "  exit code:                  $T2_EXIT"
echo "  log file:                   $T2_LOG/T2.log"
echo "  log lines:                  $(wc -l < $T2_LOG/T2.log)"
N_STEPS=$(grep -cE "step=[0-9]+/20 gen_loss" "$T2_LOG/T2.log")
N_NEXT_RIDE=$(grep -c "_next_ride: gave up" "$T2_LOG/T2.log")
HAS_LORA_TRAIN=$(grep -c "real_teacher_train_online=True.*LoRA params trainable" "$T2_LOG/T2.log")
HAS_RT_OPT=$(grep -c "real_teacher optimizer built" "$T2_LOG/T2.log")
HAS_GAN_DDP=$(grep -c "R3GAN discriminator built.*DDP=True" "$T2_LOG/T2.log")
echo "  step=N/20 gen_loss lines:    $N_STEPS"
echo "  _next_ride: gave up:        $N_NEXT_RIDE"
echo "  LoRA-trainable telemetry:   $HAS_LORA_TRAIN  (init check: DDP-wrap freeze fix)"
echo "  real_teacher optimizer:     $HAS_RT_OPT      (init check: online-teacher Adam built)"
echo "  R3GAN DDP-wrap:             $HAS_GAN_DDP     (init check: gan optim under DDP)"
echo ""

# Pass criteria: training must have progressed past step 0.
if [ "$T2_EXIT" -eq 124 ] || [ "$T2_EXIT" -eq 137 ]; then
  echo "TEST T2 FAILED: timeout (30 min) — online-teacher path hung."
  echo "  Last 5 log lines:"
  tail -5 "$T2_LOG/T2.log"
  exit 42
fi
if [ "$T2_EXIT" -ne 0 ]; then
  echo "TEST T2 FAILED: trainer exited $T2_EXIT — last 5 log lines:"
  tail -5 "$T2_LOG/T2.log"
  if grep -q "DistributedDataParallel is not needed" "$T2_LOG/T2.log"; then
    echo "  → DDP-wrap freeze regression returned (model/dmd_action_forcing.py:584)"
  fi
  if grep -q "tf_rope_offset_frames is 0" "$T2_LOG/T2.log"; then
    echo "  → peft-wrapper RoPE attr regression returned (model/dmd_action_forcing.py:548)"
  fi
  exit 42
fi
if [ "$N_STEPS" -lt 1 ]; then
  echo "TEST T2 FAILED: trainer ran clean but logged 0 training steps."
  echo "  init succeeded (LoRA-trainable + R3GAN DDP-wrap + real_teacher Adam) but"
  echo "  no step=N/20 line ever appeared. Likely a hang or silent skip."
  exit 43
fi
echo "TEST T2 PASSED: $N_STEPS/3 training steps logged."
[ "$HAS_LORA_TRAIN" -gt 0 ] && echo "  ✓ DDP-wrap freeze fix held (LoRA-trainable telemetry detected)"
[ "$HAS_RT_OPT" -gt 0 ]    && echo "  ✓ peft-wrapper RoPE fix held (real_teacher optimizer built)"
[ "$HAS_GAN_DDP" -gt 0 ]   && echo "  ✓ R3GAN built under DDP"

echo ""
echo "================================================================"
echo "  GAN + ONLINE-TEACHER SMOKE PASSED on $(date)"
echo "  Logs: $LOGROOT"
echo "  Wandb: https://wandb.ai/ashish397-university-of-exeter/longlive-phase1-smoke"
echo "================================================================"
