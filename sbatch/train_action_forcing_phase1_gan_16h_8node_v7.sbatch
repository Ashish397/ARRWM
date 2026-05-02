#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-gan-v7-flash-dmd
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing DMD + R3GAN — v7 (Flash-DMD priorities 1+2).
#
# Migration of the Flash-DMD paper (arXiv:2511.20549, Tencent / SJTU)
# decoupling strategy onto our existing single-teacher (LoRA-only)
# DMD pipeline. Two changes vs v6:
#
#   1. Timestep-aware DMD/GAN gating.
#      ``flash_dmd_split_timestep: 500``
#      Each gen iter rolls one DDP-synced scalar t ~ U[0, 1000):
#        * t > 500 (high-noise): DMD active (per-frame timesteps
#          constrained to [500, 1000]); generator's GAN loss is
#          SKIPPED. Discriminator updates as normal.
#        * t <= 500 (low-noise): DMD compute SKIPPED entirely (zero
#          loss, no DDP all-reduce on real_score gradients);
#          generator's GAN loss is the sole signal.
#      The aux-teacher pass is unaffected — it continues to fire on
#      every gen iter when ``aux_teacher_loss_weight > 0``.
#      Why: paper §3.3 — DMD is informative at high noise (global
#      structure), GAN is informative at low noise (texture, realism).
#      Summing them every iter creates conflicting gradients that
#      slow convergence (DMD2's known weakness).
#
#   2. EMA fake_score from generator.
#      ``fake_score_ema_weight: 0.95``
#      After each generator optimizer.step(), the fake_score's params
#      get EMA-pulled toward the generator's newly-updated params:
#          ψ ← 0.95 · ψ + 0.05 · θ
#      This lets fake_score track p_gen with much fewer dedicated
#      diffusion-loss updates, replacing DMD2's TTUR=5 burden.
#      Paired with ``dfake_gen_update_ratio: 1`` (vs v6's 5) the
#      critic still sees enough diversity but the wallclock per
#      training iter drops since 4 of every 5 critic-only iters
#      are eliminated.
#      Why: paper §3.3 — DMD2 used TTUR=5 because the fake_score
#      doubled as a discriminator. Our fake_score is already
#      separated (R3GAN is a distinct module), so the EMA strategy
#      is a strict improvement.
#
# Inherited from v6 (unchanged):
#   * Single-teacher mode (``dmd_frozen_teacher_pass_enabled=false``):
#     DMD scoring + aux pass both go through the LoRA real_score.
#   * CFG=1, causal_mask=false (bidir), real_teacher_lr=5e-5,
#     warmup=200, vae_roundtrip=true, dmd_context_mix_p=0.15,
#     real_teacher_input_source=mix(p=0.5),
#     aux_teacher_loss_weight=1.0.
#
# Smoke validation:
#   srun --jobid=<existing-bash> --overlap bash sbatch/smoke_run_attached.sh
#   (smoke runner needs ``flash_dmd_split_timestep=500
#    fake_score_ema_weight=0.95 dfake_gen_update_ratio=1`` overrides
#    to mirror v7; edit before launch.)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_v7
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
echo "Starting Phase-1 Action-Forcing DMD + R3GAN v7 (Flash-DMD) on $(date)"
echo "  v7 deltas vs v6: flash_dmd_split_timestep=500, fake_score_ema_weight=0.95,"
echo "                   dfake_gen_update_ratio=1 (down from 5 — EMA replaces TTUR=5 cadence)."

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
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    max_rolls_per_ride=300 \
    dfake_gen_update_ratio=1 \
    model_kwargs.local_attn_size=42 \
    num_workers=0 \
    boundary_vae_roundtrip=true \
    collapse_mae_threshold=0.2 \
    mae_extension_threshold=0.2 \
    local_attn_size_schedule=[[0,42]] \
    dmd_context=mix \
    dmd_context_mix_p=0.15 \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=1.0 \
    real_teacher_lr=5e-5 \
    real_teacher_warmup_steps=200 \
    real_teacher_input_source=mix \
    real_teacher_input_mix_gt_p=0.5 \
    dmd_frozen_teacher_pass_enabled=false \
    aux_teacher_loss_weight=1.0 \
    flash_dmd_split_timestep=500 \
    fake_score_ema_weight=0.95 \
    run_name=action_forcing_phase1_gan_j${SLURM_JOB_ID}_v7_flash_dmd

echo "Training completed on $(date)"
