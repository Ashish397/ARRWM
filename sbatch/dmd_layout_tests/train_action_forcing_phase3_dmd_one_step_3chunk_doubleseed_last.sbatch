#!/bin/bash
#SBATCH --job-name=af-p3-dmd-3chunk-doubleseedlast
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=01:30:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-3 DMD one-step — single frozen teacher, no aux pass.
#
# Cleanup pass on top of phase_3_refactor (dmd_one_step branch):
#   * dmd_context hardcoded to "self".
#   * Single teacher (merged v14, frozen). No online LoRA training,
#     no aux pass, no dual-teacher.
#   * SAM-pixel GAN + LatentSAM2Critic removed.
#   * MANIQA + perceptual approx surrogates removed.
#   * 14B foreign_real_teacher path removed.
#   * warm_start_init removed.
#   * teacher_freeze_detect removed.
#
# GAN OFF, 21-frame window, 200-step max.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase3_dmd.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_dmd_one_step
WANDB_SAVE_DIR=wandb

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

# GAN OFF -> no SAM2 prereq checks needed.

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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "Starting Phase-3 DMD ONE-STEP (v14-faithful asymmetric) on $(date)"
echo "  * dmd_asymmetric_scoring: ON (clean=21 GT chunks, noisy=1 student chunk)"
echo "  * Supervise the FIRST rolled chunk only; cf=18 (6-chunk seed),"
echo "    teacher clean context = GT[0:21] (7 chunks); rope_offset=21."
echo "  * 1 roll per ride (fresh seed each step)."
echo "  * GAN: OFF"
echo "  * Teacher: single frozen merged-v14 (no online, no aux pass)."
echo "  * wandb_project=longlive-phase-3."

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
    wandb_project=longlive-phase-3 \
    max_steps=200 \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    max_rolls_per_ride=1 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=21 \
    num_workers=0 \
    sample_fps=16 \
    boundary_vae_roundtrip=false \
    local_attn_size_schedule=[[0,21]] \
    real_teacher_causal_mask=false \
    real_guidance_scale=0.0 \
    dmd_context_clean_frames=18 \
    dmd_loss_weight=1.0 \
    dmd_loss_start_step=0 \
    dmd_loss_warmup_steps=20 \
    dmd_normalization_enabled=true \
    dmd_asymmetric_scoring_enabled=false \
    dmd_42f_enabled=true \
    dmd_42f_2chunk=true \
    dmd_42f_seed_last=true \
    dmd_42f_gt_after_chunks=2 \
    dmd_only_first_chunk_per_ride=true \
    flash_dmd_enabled=false \
    fake_score_ema_weight=0.95 \
    gan_enabled=false \
    gan_loss_weight=0.0 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=dmd_one_step_3chunk_doubleseed_last_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
