#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-gan-v12-fd-sobolev
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
# Phase-1 Action-Forcing DMD + R3GAN — v12 (FD Sobolev; 3h timebox).
#
# v12 deltas vs v10:
#
#   * Finite-difference Sobolev (option 2): train the critic's input-
#     gradient to match the disc's input-gradient WITHOUT graph-on
#     V+SAM2 backward. For each gen step, perturb fake_lat by
#     ε ~ N(0, 0.01²) on a 1-frame subset (no_grad), forward V+SAM2
#     on the perturbed point to get disc(decode(z+ε)), compute
#     finite-difference target ``Δ_disc = disc(z+ε) - disc(z)``, and
#     train critic to match the corresponding ``Δ_critic = critic(z+ε)
#     - critic(z)``. Gives the critic a directional-derivative signal
#     that matches the disc, without needing to backprop through V+SAM2
#     (which OOMs on a single-frame subset, see v8 docstring).
#     ``gan_critic_fd_loss_weight=1.0`` adds the FD term to L_critic.
#
#   * Critic stays at v10 size (24M, d_model=512, num_blocks=6) and
#     uses single-step training (gan_critic_updates_per_step=1) so
#     v12 is a clean test of FD Sobolev's effect alone, not bundled
#     with multi-step / multi-noise / bigger-critic. v11 tests those
#     options separately.
#
#   * gan_warmup_steps=400 (was 1000) — at v10's ~65 sec/step + 3h
#     budget we get ~165 steps; with ratio=5 effective gen steps are
#     33 — plenty for the GAN to ramp to full strength before run end.
#
#   * gan_critic_warmup_steps=100 (was 150) — Path 3 fires at step
#     100 so we have ~65 steps of GAN-influencing gen training.
#
# Inherited from v10 (the working stack):
#
#   * R1/R2 GAP-unscaled penalty (disc properly regularized).
#   * dfake_gen_update_ratio=5 (v6-style cadence).
#   * real_guidance_scale=1.0.
#   * gan_sam2_distilled_critic=true, INLINE Paths 1+2 before Path 3.
#
# v10 deltas vs v9:
#
#   * R1/R2 GAP normalization fix: feature-manifold gradient penalty
#     was effectively zero in v9 because the ADM heads' GAP averaging
#     shrinks per-feature gradients by 1/(H*W). v10 multiplies each
#     scale's per-sample squared-norm by H*W to undo the GAP shrinkage.
#     Result: R1/R2 land at paper magnitude (~0.01-1.0 with γ=1.0)
#     and the disc is properly regularized — addresses the v9 disc
#     saturation / vanishing gradient that would have killed Path 3.
#
#   * dfake_gen_update_ratio=5 (v6-style cadence). Fake_score (DMD
#     critic) trains every iter; gen + disc + latent critic train every
#     5 iters. v6 used this and the natural generator quality was
#     strong. Side-effect: disc/critic also fire on gen iters only,
#     so they get 1/5 the updates per wallclock — compensated by
#     shorter Path 3 warmup (gan_critic_warmup_steps=150 -> 30
#     effective disc updates before gen-side GAN turns on).
#
#   * real_guidance_scale=1.0 (was 2.0). v9's CFG=2 may have amplified
#     teacher instability; reverting to v6's CFG=1 to align with the
#     "natural generator" baseline that was visually strong.
#
#   * gan_critic_warmup_steps=150 (was 500). With ratio=5 and properly
#     regularized R1/R2, the disc is ready by step ~150 (= 30 disc
#     updates). Faster Path 3 fire = faster signal to the gen.
#
# Inherited from v9 (distilled critic stack):
#   * gan_sam2_distilled_critic=true — LatentSAM2Critic (24M params)
#     receives the gen-side GAN gradient. V+SAM2 never appears in
#     the gen's autograd graph. Sobolev OFF (value-only distillation).
#   * INLINE Paths 1+2 before Path 3 — gen consumes a freshly-stepped
#     critic each iter (action-critic-style boost).
#   * gan_sam2_resolution=512, pad_to_square=true (480x832 → 832x832
#     → 512x512, geometry preserved).
#   * Generator + real_score + fake_score gradient checkpointing.
#   * real_teacher_train_online=true, aux_teacher_loss_weight=1.0
#     (online LoRA flow teacher feeding aux loss into gen).
#
# Inherited from v8:
#   * flash_dmd_paper_aligned_adv=true (dual_grad rollout — paper §3.3
#     Eq. 8-9 with K/V decoupling).
#   * gan_loss_weight=1.0, gan_warmup_steps=1000.
#   * Flash-DMD timestep gating (split_timestep=500), fake_score EMA.
#   * dmd_context=mix, dmd_context_mix_p=0.15.
#   * causal_mask=false (bidir), boundary_vae_roundtrip=true.
#
# PREREQUISITES (must complete BEFORE submitting):
#
#   1. Install SAM2 into the arrwm conda env:
#        source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
#        pip install git+https://github.com/facebookresearch/sam2.git
#
#   2. Download SAM2.1 Hiera-B+ checkpoint (~150 MB):
#        mkdir -p /scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints
#        cd /scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints
#        wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
#
#   3. Locate the matching config (ships with the sam2 package):
#        python -c "import sam2, pathlib; print(pathlib.Path(sam2.__file__).parent / 'configs' / 'sam2.1' / 'sam2.1_hiera_b+.yaml')"
#      Symlink or copy that into ./sam2_checkpoints/, OR update
#      ``GAN_SAM2_CFG`` below to point at the package path directly.
#
# Pre-flight checks below fail-fast if any of the three are missing.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_v12_fd_sobolev
WANDB_SAVE_DIR=wandb

# ---- SAM2 paths (must be set; see PREREQUISITES) ----
GAN_SAM2_CKPT=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_base_plus.pt
GAN_SAM2_CFG=/scratch/u6ex/as1748.u6ex/ARRWM/sam2_checkpoints/sam2.1_hiera_b+.yaml

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

# Pre-flight SAM2 sanity checks. Fail-fast on missing prereqs so the
# job exits at submission rather than ~5 min into model load.
python -c "import sam2" 2>/dev/null || {
  echo "ERROR: sam2 package not installed. See PREREQUISITES in this sbatch."
  exit 1
}
[ -f "$GAN_SAM2_CKPT" ] || {
  echo "ERROR: SAM2 checkpoint not found at $GAN_SAM2_CKPT"
  exit 1
}
[ -f "$GAN_SAM2_CFG" ] || {
  echo "ERROR: SAM2 config not found at $GAN_SAM2_CFG"
  exit 1
}

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
echo "GAN_SAM2_CKPT=$GAN_SAM2_CKPT"
echo "GAN_SAM2_CFG=$GAN_SAM2_CFG"
echo "Starting Phase-1 Action-Forcing DMD + R3GAN v12 (FD Sobolev) on $(date)"
echo "  v12 deltas vs v10: gan_critic_fd_loss_weight=1.0 (finite-difference Sobolev),"
echo "                     gan_critic_fd_sigma=0.01, fd_frames=1, fd_n_directions=1,"
echo "                     critic stays at v10 size (24M, d_model=512, num_blocks=6),"
echo "                     gan_warmup_steps=400 (was 1000), gan_critic_warmup_steps=100 (was 150)."

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
    flash_dmd_split_timestep=500 \
    fake_score_ema_weight=0.95 \
    flash_dmd_paper_aligned_adv=true \
    gan_loss_weight=1.0 \
    gan_warmup_steps=400 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=512 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=100 \
    gan_critic_grad_frames=1 \
    gan_critic_grad_full_every=0 \
    gan_critic_grad_loss_weight=0.0 \
    gan_critic_fd_loss_weight=1.0 \
    gan_critic_fd_sigma=0.01 \
    gan_critic_fd_frames=1 \
    gan_critic_fd_n_directions=1 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=action_forcing_phase1_gan_j${SLURM_JOB_ID}_v12_fd_sobolev

echo "Training completed on $(date)"

