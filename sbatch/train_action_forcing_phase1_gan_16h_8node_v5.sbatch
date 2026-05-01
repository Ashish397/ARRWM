#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-gan-v5-dual-teacher
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
# Phase-1 Action-Forcing DMD + R3GAN — v5 (dual-teacher).
#
# Effective config = action_forcing_phase1_gan.yaml + the explicit
# overrides below. The override block is the source of truth for this
# RUN — the comment list further down documents what's set there and
# why. Knobs not overridden inherit from the overlay verbatim.
#
# Run-defining knobs (set in --override below):
#
#   real_teacher_train_online: true        Online v14 LoRA stays
#                                          unmerged + trainable.
#   real_teacher_causal_mask:  false       Bidirectional joint TF
#                                          mask on real_score (vs v14
#                                          parity true). Used for the
#                                          aux teacher pass; the
#                                          frozen merged-v14 inherits
#                                          this too — accept the
#                                          step-1 distribution
#                                          mismatch from the warm-
#                                          start.
#   real_teacher_lr:           5.0e-05     v14 default LR; warmup
#                                          cushions the first 200
#                                          steps.
#   real_teacher_warmup_steps: 200         Linear warmup on the LoRA
#                                          optimizer.
#   real_guidance_scale:       1.0         CFG=1 (no extrapolation).
#                                          The CFG=3 path was found
#                                          to be unstable when the
#                                          uncond branch drifts under
#                                          the LoRA training; CFG=1
#                                          short-circuits the
#                                          (cond − uncond) path.
#   real_teacher_input_source: "mix"       Aux pass picks per-iter
#                                          Bernoulli — half iters
#                                          add_noise(student, ε, t),
#                                          half add_noise(GT, ε, t).
#                                          DDP-synced.
#   real_teacher_input_mix_gt_p: 0.5
#   dmd_frozen_teacher_pass_enabled: true  Dual-teacher mode. Hold a
#                                          frozen merged-v14 ALONG-
#                                          SIDE the LoRA-online
#                                          real_score. DMD scoring
#                                          forwards through the
#                                          frozen one with self-only
#                                          clean_x (proper p_real,
#                                          no GT contamination, no
#                                          moving target). The aux
#                                          pass forwards through the
#                                          LoRA with GT-mixed clean_x
#                                          and flows gradient back to
#                                          the student via the noise
#                                          base.
#   aux_teacher_loss_weight:   1.0         Weight on the aux-pass
#                                          loss in the gen-step total
#                                          (0.0 disables aux pass).
#   dmd_context:               "mix"       Uniform linear blend on
#                                          clean_x across all frames.
#   dmd_context_mix_p:         0.15        0.85*self + 0.15*GT
#                                          (yinyang_random was
#                                          removed; chunk-level dice
#                                          had introduced disjoint
#                                          artefacts in the student's
#                                          reconstruction).
#   boundary_vae_roundtrip:    true        Re-enabled — the bf16/fp32
#                                          conv crash from earlier
#                                          smokes is no longer
#                                          firing.
#
# Inherited from the overlay (unchanged here):
#   gan_*                                   R3GAN recipe (loss_weight=
#                                           0.05, R1=R2=1.0, lr=2e-4,
#                                           betas=(0,0.9), 500-step
#                                           linear warmup, 1:1 D:G).
#   local_attn_size_schedule:  [[0, 42]]
#   mae_extension_threshold:   0.2
#   collapse_mae_threshold:    0.2
#
# Dropped in v5 vs v4:
#   * aux_student_gt_mse_loss_weight — REMOVED ENTIRELY (mean-seeking
#     latent L2 is a regression-to-blur signal in DMD; the GAN already
#     fights blur on the student side, and the online teacher carries
#     the GT-pull through the (fake-real) gradient).
#   * dmd_context_yinyang / dmd_context_yinyang_random — REMOVED. Mix
#     mode is now a uniform linear blend across all frames, fully
#     determined by dmd_context_mix_p alone.
#
# Smoke validation:
#   sbatch sbatch/smoke_action_forcing_phase1_gan_online_teacher.sbatch
#   (or srun --jobid=<existing-bash> --overlap bash sbatch/smoke_run_attached.sh)
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_v5
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
echo "Starting Phase-1 Action-Forcing DMD + R3GAN v5 (dual-teacher) on $(date)"
echo "  Dual-teacher: frozen merged-v14 in DMD path + LoRA in aux pass."
echo "  CFG=1.0, causal_mask=false (bidir), real_teacher_lr=5e-5, warmup=200, vae_roundtrip=true."
echo "  dmd_context_mix_p=0.15, real_teacher_input_source=mix(p=0.5)."

srun --ntasks-per-node=1 bash -c \
  'echo "=== $(hostname) GPU memory ==="; nvidia-smi --query-gpu=index,memory.total,memory.free,memory.used --format=csv,noheader 2>/dev/null || echo "nvidia-smi not available"' \
  || true
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_NNODES"
echo "GPUs per node: $SLURM_GPUS_ON_NODE"
echo "Total GPUs: $(( SLURM_NNODES * SLURM_GPUS_ON_NODE ))"
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# Overrides kept here are ONLY the production-vs-base deltas that
# don't belong in the GAN overlay (they're job-shape choices, not
# experiment knobs). All experiment knobs come from the overlay.
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
    dmd_frozen_teacher_pass_enabled=true \
    aux_teacher_loss_weight=1.0 \
    run_name=action_forcing_phase1_gan_j${SLURM_JOB_ID}_v5_dual_teacher


echo "Training completed on $(date)"
