#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-aux-10h
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
# Phase-1 Action-Forcing DMD WITH AUX LOSSES (action critic + frozen
# motion teacher z-guidance). 1 node x 4 GPUs, 10h.
#
# Same CF chunkwise DMD recipe as the base
# (sbatch/train_action_forcing_phase1_10h.sbatch / configs/
# action_forcing_phase1.yaml) PLUS the auxiliary action critic
# training loop:
#
#   Per generator step:
#     1. Roll the student through 21 frames + (optional) MAE
#        extensions (wandb metrics: ``gen/baseline_last_chunk_mae``,
#        ``gen/last_chunk_mae``, ``gen/mae_extension_count``).
#     2. Compute CF-parity DMD loss against the v14 LoRA-merged
#        real_score and the generator-mirrored fake_score.
#     3. Decode pred_x0 to pixels via the frozen WanVAE and run
#        CoTracker-3 (offline, grid=10, fp16 autocast) → ss_vae
#        encoder + tanh-squash to produce the chunkwise teacher z
#        ([B, 7, 8] for the 21-frame rollout).
#     4. Run ``critic_updates_per_step=2`` independent AdamW steps on
#        the action critic (lr=3e-4, weight=0.25) against the teacher
#        z target.
#     5. Frozen critic forward on the LIVE pred_x0 + add a z2/z7 MSE
#        guidance loss (weight=0.125, linear warmup 500 steps from
#        ``warmup_steps``=0) to the DMD loss before generator
#        backward.
#     6. Generator backward + EMA update; fake_score backward (1:1
#        DMD2 update cadence).
#
#   Wandb metrics (under the ``gen/`` namespace, prefixed with
#   ``train/`` from the critic helper):
#     train/critic_z_loss          (per-step critic training loss)
#     train/critic_loss            (= weight * critic_z_loss)
#     train/critic_z2_mse          (per-dim teacher MSE, z2)
#     train/critic_z7_mse          (per-dim teacher MSE, z7)
#     train/gen_z_loss             (gen-side guidance MSE; 0 when
#                                   guidance_scale=0 i.e. inside warmup)
#     train/gen_action_loss        (= guidance_scale * gen_z_loss;
#                                   what's actually added to DMD loss)
#     train/teacher_z2_mean        (teacher z2 mean; sanity check)
#     train/teacher_z7_mean        (teacher z7 mean)
#     train/z_guidance_scale       (current ramped guidance scale)
#     train/teacher_unavailable    (1.0 on iters where the teacher
#                                   forward failed and aux was
#                                   silently skipped — track this!)
#
# Memory cost vs the base config: ~2-3 GB for cotracker+ss_vae per
# rank, plus the critic params (~few hundred MB). Per-iter
# wallclock: cotracker forward dominates (~150ms / iter on H100 at
# grid=10), so expect ~30-60% throughput hit vs the pure-DMD base.
#
# To run the pure-DMD A/B (base config) instead:
#   sbatch sbatch/train_action_forcing_phase1_10h.sbatch
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_aux.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_aux_10h
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
echo "Starting phase-1 Action-Forcing DMD + AUX (1 node, 4 GPUs, 10h) on $(date)"

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
    run_name=action_forcing_phase1_aux_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
