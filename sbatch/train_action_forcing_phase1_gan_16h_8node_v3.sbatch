#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-gan-v3-yinyang-det-15gt
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
# Phase-1 Action-Forcing DMD + R3GAN — v3 (yinyang deterministic, 15%
# GT splash, lower real_teacher LR, no VAE roundtrip).
#
# This is a snapshot of the v3 experiment configuration. Sister files
# (train_action_forcing_phase1_gan_16h.sbatch + the _v2 variant) are
# preserved as records of earlier configurations.
#
# Diff vs v2 (action_forcing_phase1_gan_16h_8node_v2.sbatch):
#
#   * dmd_context_mix_p:        0.5  → 0.15
#       (15% GT splash, 85% self — was 50/50)
#   * dmd_context_yinyang_random:           ADDED = false
#       (NEW knob: skips the per-chunk Bernoulli shuffle inside
#        _resolve_dmd_context_mask. Every chunk now blends with the
#        SAME 0.85*self + 0.15*GT weights instead of randomly choosing
#        each chunk to be GT-dominant or self-dominant. Removes
#        per-iter chunk-level variance from real_score's conditioning.)
#   * real_teacher_lr:          5e-5 → 1e-5
#       (5× lower: parity with the generator's LR; calmer LoRA
#        movement now that warmup is also off)
#   * real_teacher_warmup_steps:           ADDED = 0
#       (was 200 in phase1.yaml; explicitly pinned to 0 here for
#        traceability)
#   * real_teacher_causal_mask: true → false
#       (matches gan.yaml's bidirectional contract — same as the
#        smoke harness; v2's true was an override of gan.yaml's
#        intent that we're undoing for this experiment)
#   * boundary_vae_roundtrip:              ADDED = false
#       (the chunk-boundary VAE roundtrip in dmd_action_forcing.py
#        crashes with "Input type (BFloat16) and bias type (float)
#        should be the same" because the VAE Conv3d kernels are
#        fp32 and the streaming latents are bf16. Other phase1
#        variants (freeze/scdmd/aux_*) already disable this; we
#        match. Re-enable as a quality-tuning task once the VAE
#        call wraps the input/output in proper fp32↔bf16 casts.)
#
# All other v2 knobs preserved: action_teacher_mode=off,
# action_critic_aux_enabled=false, state_probe_aux_enabled=false,
# max_rolls_per_ride=300, dfake_gen_update_ratio=5,
# collapse_mae_threshold=0.2, mae_extension_threshold=0.2,
# model_kwargs.local_attn_size=42, local_attn_size_schedule=[[0,42]],
# dmd_context=mix, dmd_context_yinyang=true, real_teacher_train_online=
# true, real_guidance_scale=1.0, num_workers=0.
#
# Smoke validation (run before this sbatch):
#   sbatch sbatch/smoke_action_forcing_phase1_gan_online_teacher.sbatch
#   (or via srun --jobid=<bash> --overlap into an existing allocation)
# Bug coverage validated by the smoke series:
#   1. DDP-wrap freeze regression on real_score
#   2. peft-wrapper RoPE attribute forwarding (tf_rope_offset_frames)
#   3. Activation checkpointing on real_score's WanBlocks (memory)
#   4. cond-dict graph leak between critic.backward() and rt_loss.backward()
#   5. VAE Conv3d bf16/fp32 mismatch (this is why boundary_vae_roundtrip=false)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_v3_yinyang_det_15gt
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
echo "Starting Phase-1 Action-Forcing DMD + R3GAN v3 on $(date)"
echo "  v3 deltas: yinyang_random=false, mix_p=0.15, real_teacher_lr=1e-5, boundary_vae_roundtrip=false"

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
    dfake_gen_update_ratio=5 \
    collapse_mae_threshold=0.2 \
    mae_extension_threshold=0.2 \
    model_kwargs.local_attn_size=42 \
    local_attn_size_schedule=[[0,42]] \
    dmd_context=mix \
    dmd_context_mix_p=0.15 \
    dmd_context_yinyang=true \
    dmd_context_yinyang_random=false \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=3.0 \
    real_teacher_lr=1e-5 \
    real_teacher_warmup_steps=0 \
    boundary_vae_roundtrip=false \
    num_workers=0 \
    run_name=action_forcing_phase1_gan_j${SLURM_JOB_ID}_v3_yinyang_det_15gt_lr1e5

echo "Training completed on $(date)"
