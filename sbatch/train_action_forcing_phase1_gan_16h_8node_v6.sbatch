#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-gan-v6-lora-only
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
# Phase-1 Action-Forcing DMD + R3GAN — v6 (LoRA-only teacher).
#
# Diff vs v5 (action_forcing_phase1_gan_16h_8node_v5.sbatch):
#
#   * dmd_frozen_teacher_pass_enabled: true → false
#       Drops the frozen merged-v14 teacher entirely. v5 used it as
#       the DMD oracle (proper p_real, no GT contamination); empirically
#       that teacher hurt the run while the LoRA-side aux pass
#       (pred_real_lora video) was producing strong results. v6
#       collapses to single-teacher mode: DMD scoring forwards
#       through ``self.real_score`` (the LoRA, with GT-mixed clean_x);
#       aux pass continues to fire on the same module with grad on
#       LoRA + student.
#
#   * Memory: drops ~5 GB per rank (no frozen 14B-bit copy held).
#
# Effective per-iter teacher work:
#   * DMD pass (every gen iter): forward ``self.real_score`` (LoRA,
#     no_grad) on the same noisy student input as fake_score, with
#     GT-mixed clean_x. (fake - real) gradient drives the student.
#   * Aux pass (every gen iter, undetached): forward LoRA with grad,
#     compute (flow_pred - (ε - GT))² → trains LoRA, also flows back
#     to student via the noise base when input_source picks the
#     student.
#   * Post-critic teacher step (every iter, detached): trains LoRA
#     against GT on detached chunk. Provides the every-iter LoRA
#     training cadence (vs the every-5-iter aux-pass cadence at
#     dfake_gen_update_ratio=5).
#
# All other v5 overrides preserved:
#   * CFG=1, causal_mask=false (bidir), real_teacher_lr=5e-5,
#     warmup=200, vae_roundtrip=true, dmd_context_mix_p=0.15,
#     real_teacher_input_source=mix(p=0.5), aux_teacher_loss_weight=1.0.
#
# Smoke validation:
#   srun --jobid=<existing-bash> --overlap bash sbatch/smoke_run_attached.sh
#   (smoke runner inherits dmd_frozen_teacher_pass_enabled from the
#    overlay = true; to mirror v6, override to false in the runner or
#    re-test by editing the runner's --override block.)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_v6
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
echo "Starting Phase-1 Action-Forcing DMD + R3GAN v6 (LoRA-only teacher) on $(date)"
echo "  v6 deltas vs v5: dmd_frozen_teacher_pass_enabled=false (drop frozen merged-v14)."
echo "  Single-teacher: DMD scoring + aux pass both go through self.real_score (LoRA)."
echo "  CFG=1.0, causal_mask=false (bidir), real_teacher_lr=5e-5, warmup=200, vae_roundtrip=true."

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
    run_name=action_forcing_phase1_gan_j${SLURM_JOB_ID}_v6_lora_only

echo "Training completed on $(date)"
