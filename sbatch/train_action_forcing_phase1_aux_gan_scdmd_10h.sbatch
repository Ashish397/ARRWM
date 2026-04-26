#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-aux-gan-scdmd-10h
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
# Phase-1 Action-Forcing DMD WITH AUX LOSSES + R3GAN + SC-DMD.
# 1 node x 4 GPUs, 10h.
#
# Same recipe as ``train_action_forcing_phase1_aux_gan_10h.sbatch``
# (CF chunkwise DMD + 21-frame action-forcing rollout + infinity-RoPE
# + MAE extensions + action critic + frozen motion teacher z-guidance
# + R3GAN discriminator) PLUS the SC-DMD (Salt) semigroup defect
# regularizer.
#
# SC-DMD reference:
#   "Salt: Self-Consistent Distribution Matching with Cache-Aware
#    Training for Fast Video Generation" (arXiv 2604.03118v1)
#
# What it does:
#   DMD trains the student to match the teacher's score field at
#   each timestep INDEPENDENTLY. At inference the student composes
#   multi-step denoising trajectories — and that's where DMD students
#   drift, because per-timestep matching doesn't guarantee that
#   composing two consecutive matches lands on the same point as a
#   direct match. SC-DMD adds an explicit regularizer:
#
#     L_SC = E[|| Ψ_θ^{ts→te}(x_ts)
#                - Ψ_θ^{tm→te}(Ψ_θ^{ts→tm}(x_ts)) ||²]
#
#   penalising the gap between the direct one-step path
#   (``t_s → t_e``) and the composed two-step path
#   (``t_s → t_m → t_e``). The Salt paper proves this is a local
#   convergence regularizer (Theorem 1) and shows ~0.5-1.0 dB FID
#   gains on FastVideo with negligible compute cost.
#
# Implementation (this run):
#   * Single chunk (chunk-0, fresh KV cache) per gen step.
#   * 2 extra DiT forwards on a 3-frame chunk per gen step.
#   * Triplet ``(t_s, t_m, t_e)`` sampled uniformly from
#     ``denoising_step_list`` at each step (rank-broadcast for DDP
#     lockstep).
#   * Linear weight ramp 0 → ``sc_dmd_loss_weight`` over
#     ``sc_dmd_warmup_steps`` (default 500). SC defect is added to
#     the unified generator backward alongside DMD/aux/GAN.
#
# Per generator step:
#   1. Roll the student through 21 frames + (optional) MAE extensions.
#   2. CF-parity DMD loss against v14 LoRA-merged real_score and the
#      generator-mirrored fake_score.
#   3. Run cotracker → ss_vae → teacher z (chunkwise [B,7,8]).
#   4. Train action critic (2 indep AdamW steps, lr=3e-4, weight=0.25).
#   5. Frozen-critic z-guidance (weight=0.125 ramped over 500 steps).
#   6. R3GAN D-update (RpGAN-D + R1 + R2 grad penalties).
#   7. R3GAN G-update term (frozen D forward on pred_image, RpGAN-G
#      added to gen loss; ramped over 500 steps via gan_warmup_steps).
#   8. SC-DMD pass (2 forwards on chunk 0, fresh small KV cache;
#      semigroup defect added to gen loss; ramped over 500 steps via
#      sc_dmd_warmup_steps).
#   9. Generator backward + EMA; fake_score backward.
#
# Wandb metrics added vs aux+gan config (under ``gen/`` namespace):
#   gen/train/sc_dmd_t_s              triplet rung high (noise level)
#   gen/train/sc_dmd_t_m              triplet rung mid
#   gen/train/sc_dmd_t_e              triplet rung low (target)
#   gen/train/sc_dmd_loss_raw         raw SC defect MSE (pre-weight)
#   gen/train/sc_dmd_loss_weighted    what's actually added to gen loss
#   gen/train/sc_dmd_weight_effective current ramped weight
#   gen/train/sc_dmd_skipped          1 if skipped (denoising_step_list<3)
#
# Health checks:
#   * ``sc_dmd_loss_raw`` should drop O(10x) over the first ~1000
#     steps as the velocity field becomes self-consistent. If it
#     stays flat → ``sc_dmd_loss_weight`` is too small (or ramp
#     too slow); bump to 0.1.
#   * ``sc_dmd_loss_raw`` should NOT crash to 0 — that means the
#     student has collapsed to a degenerate point (constant output);
#     drop ``sc_dmd_loss_weight`` to 0.02 and check DMD's own loss.
#   * Triplet rung distribution should be uniform over
#     ``denoising_step_list``; if one rung dominates it indicates
#     a sampling bug. (Statistical check — run a few hundred steps,
#     verify all rungs see nontrivial mass.)
#
# Memory cost vs aux+gan config: ~10-20 MB extra (small fresh KV
# cache for SC pass; activations dwarfed by main rollout).
# Per-iter wallclock: +30-50ms for the 2 SC forwards (1-2% overhead
# vs aux+gan baseline).
#
# To DISABLE SC-DMD (regress to AUX+GAN-only), edit the config
# (``sc_dmd_enabled: false``) — the trainer skips the SC pass.
#
# Sibling runs (4-way A/B):
#   * sbatch/train_action_forcing_phase1_10h.sbatch          (DMD only)
#   * sbatch/train_action_forcing_phase1_aux_10h.sbatch       (DMD + aux)
#   * sbatch/train_action_forcing_phase1_aux_gan_10h.sbatch   (DMD + aux + GAN)
#   * sbatch/train_action_forcing_phase1_aux_gan_scdmd_10h.sbatch (THIS — DMD + aux + GAN + SC)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_aux_gan_scdmd.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_aux_gan_scdmd_10h
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
echo "Starting phase-1 Action-Forcing DMD + AUX + R3GAN + SC-DMD (1 node, 4 GPUs, 10h) on $(date)"

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
    run_name=action_forcing_phase1_aux_gan_scdmd_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
