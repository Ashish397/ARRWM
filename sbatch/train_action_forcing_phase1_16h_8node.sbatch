#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-16h-8node_0cfg
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-1 Action-Forcing DMD — DEFAULTS, 1 node x 4 GPUs, 16h.
#
# Pure DMD baseline: no aux losses, no R3GAN, no SC-DMD. The defaults
# below come from configs/action_forcing_phase1.yaml. Notable
# behaviour (post Phase-A + Phase-B work):
#
#   * dmd_context: "GT" (default) — real_score sees the noised-GT
#     clean half during DMD scoring (clean_x_aug_t=20 noise level for
#     symmetry-breaking). fake_score's clean_x is always the
#     student-rolled "self" view. Set dmd_context: "self" for the
#     A/B variant where real also sees the self view.
#   * dmd_context_clean_frames: 9 — KV-cache seed prefill = 9 GT
#     frames at t=0 before each rollout. Single source of truth on
#     the model; the trainer reads it back.
#   * Random ride offset s ~ U[0, ride_len/2] per iter, MIN-reduced
#     across DDP ranks + broadcast from rank 0 so all ranks's
#     all_reduces stay in lockstep.
#   * No-grad finish-denoise past the random exit rung (LongLive-
#     style): chunks exit at a randomly-sampled rung for the gradient
#     signal, then continue under no_grad to the last rung so the
#     KV cache K/V is committed from a fully-denoised x0 estimate
#     (no noise compounding across rolling chunks).
#   * +npb leading anchor chunk per ride (model.dmd_clean_x_anchor_frames
#     = num_frame_per_block) — student rolls (rollout_frames + npb)
#     frames; clean_x_self anchor = first N (purely student-rolled,
#     no seed dip), noisy_x scoring window = last N. Locked-in
#     fundamental.
#   * Last-chunk DMD gradient mask (always on) — the last ``npb``
#     frames of every scoring window live at RoPE [N, N+shift) with
#     no clean-half counterpart in v14's TF joint sequence;
#     gradient_mask zeros them out. Applied at every DMD-score loss
#     site (gen + critic, legacy + streaming) via
#     ``model._dmd_score_grad_mask``.
#   * Bidirectional DMD scorers always full-bidir ``flash_attention``
#     (NO causal block_mask). v14/causal-student TF training paths
#     in ``wan/modules/causal_model.py`` keep their mask — only the
#     bidir scorer joint forward is unmasked.
#   * Collapse gate ON by default: ``baseline_avg_rollout_mae`` > 0.5
#     closes the streaming sequence ⇒ next iter pulls a fresh ride.
#     Threshold knob: ``collapse_mae_threshold: 0.5`` (default).
#   * Streaming-mode ON by default (LongLive persistent KV-cache +
#     variable-length chunks). The single rolling sequence spans many
#     iters with random 18-21 frame advances per iter. Disable by
#     setting ``streaming_mode: false`` in the YAML.
#   * dfake_gen_update_ratio: 2 (one gen step per two critic steps).
#   * denoising_step_list: [1000, 625, 312.5, 178.6] — 4-rung schedule.
#
# CF parity hyperparams: lr=2e-6 / 4e-7 (gen / critic), beta1=0.0,
# weight_decay=0.01, ema_weight=0.99, ema_start_step=200,
# infinity_rope=true. See the YAML header for the full alignment table.
#
# Sister sbatches (defaults + one ablation each):
#   sbatch/train_action_forcing_phase1_gan_16h.sbatch    (defaults + R3GAN)
#   sbatch/train_action_forcing_phase1_scdmd_16h.sbatch  (defaults + SC-DMD)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_16h_v4
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
echo "Starting Phase-1 Action-Forcing DMD (defaults, 1 node x 4 GPUs, 16h) on $(date)"

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# collapse_mae_threshold: 0.2 here (YAML / header doc default is 0.5).
# KV cache attention window: 42 latent frames (model_kwargs.local_attn_size
# + local_attn_size_schedule first waypoint; YAML default 21 + ramp).
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
    num_workers=0 \
    run_name=action_forcing_phase1_j${SLURM_JOB_ID}_v4

echo "Training completed on $(date)"
