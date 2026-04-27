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
# Phase-1 Action-Forcing DMD — DEFAULTS, 1 node x 4 GPUs, 10h.
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
#   * heatmap_out_noise_bar: true — drops the gen-step DMD gradient
#     on rows 0-1 of frame 0 (v14 LoRA boundary artifact).
#   * Collapse gate (ride-extension gate): MAE > 0.5 ⇒ don't extend
#     on this ride next iter. Currently single-batch is the only
#     wired path so the gate is parsed-and-stored only.
#   * dfake_gen_update_ratio: 2 (one gen step per two critic steps).
#   * denoising_step_list: [1000, 625, 312.5, 178.6] — 4-rung schedule.
#   * Streaming-mode (LongLive persistent KV-cache + variable-length
#     chunks, Phase-B): OFF by default in this YAML. Enable by adding
#     ``streaming_mode=true`` to the --override line below; that
#     activates ``setup_sequence`` / ``generate_next_chunk`` /
#     ``compute_*_loss_streaming`` and runs DMD on rolling 18-21
#     frame advances within a persistent sequence (= multi-batch
#     DMD on the same ride).
#
# CF parity hyperparams: lr=2e-6 / 4e-7 (gen / critic), beta1=0.0,
# weight_decay=0.01, ema_weight=0.99, ema_start_step=200,
# infinity_rope=true. See the YAML header for the full alignment table.
#
# To run with streaming mode on:
#   sbatch sbatch/train_action_forcing_phase1_10h.sbatch
#   # then in --override below add: streaming_mode=true streaming_max_length=57
#
# Sister sbatches (defaults + one ablation each):
#   sbatch/train_action_forcing_phase1_gan_10h.sbatch    (defaults + R3GAN)
#   sbatch/train_action_forcing_phase1_scdmd_10h.sbatch  (defaults + SC-DMD)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1.yaml
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
echo "Starting Phase-1 Action-Forcing DMD (defaults, 1 node x 4 GPUs, 10h) on $(date)"

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
