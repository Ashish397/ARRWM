#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-gan-v9-distilled-lt
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
# Phase-1 Action-Forcing DMD + R3GAN — v9 (distilled SAM2 critic + LT).
#
# v9 = v8 (SAM2-pixel GAN, paper-aligned ADM 2D heads) but with the
# pixel-space disc DISTILLED into a latent-space surrogate critic
# (LatentSAM2Critic). The generator's GAN gradient flows ONLY through
# the latent critic; V+SAM2 never appears in the gen's autograd graph.
# Architectural fix for the OOM that blocked v8 at 95 GB.
#
# Stack vs v8:
#
#   * gan_sam2_distilled_critic=true — latent-space surrogate critic
#     (LatentSAM2Critic, ~24M params: 3D conv stem + 6 transformer
#     blocks at d_model=512). Trained every iter to match the pixel
#     disc's logit values; gen gets GAN gradient from the critic.
#     Sobolev (gradient distillation) IS supported architecturally
#     but disabled here (`gan_critic_grad_loss_weight=0`) because the
#     graph-on V+SAM2 forward needed for the teacher gradient target
#     pushes us over 95 GB even at single-frame subset. Value-only
#     distillation: critic learns f's logit but not ∇_z f. Gen could
#     theoretically learn to fool the critic on values without
#     improving pixel-perceptual quality; observable in
#     ``train/critic_disc_corr`` (Pearson) vs ``train/r3gan_d_real``.
#
#   * INLINE critic training (action-critic-style boost): with Sobolev
#     off, Paths 1 (D-update) + 2 (critic value-distill) run BEFORE
#     Path 3 (gen-side ``-critic(fake)``) inside the same call to
#     ``_compute_r3gan_losses_distilled``. The generator therefore
#     consumes a FRESHLY-UPDATED critic each iter rather than the
#     previous iter's stale one — mirrors how
#     ``_compute_action_critic_losses`` trains the action critic and
#     then evaluates the gen-side guidance branch using the just-
#     stepped critic. Verifiable via ``train/critic_trained_inline=1``.
#     (When Sobolev is on, Paths 1+2 stay deferred to post-backward
#     because the graph-on V+SAM2 forward exceeds memory budget.)
#
#   * gan_sam2_resolution: 512 (paper-faithful) + pad_to_square=true.
#     The 480x832 pixel video is zero-padded (top+bottom) to 832x832
#     square BEFORE resize to (512, 512). Geometry preserved (vs
#     squash) AND Hiera pos_embed sized correctly at the square probe.
#
#   * Generator gradient checkpointing — auto-enabled when distilled
#     critic is on. Each WAN transformer block routes through
#     torch.utils.checkpoint.checkpoint; activations recomputed during
#     backward. ~33% extra compute, ~10-20 GB savings on grad-active
#     gen forwards. Required to fit dual_grad on top of distilled critic.
#
#   * real_score / fake_score gradient checkpointing — recomputes
#     teacher activations during teacher.backward instead of caching;
#     unblocks the online real_teacher LoRA + dual-grad rollout combo.
#
#   * D-update path (R1+R2+RpGAN-D) runs V+SAM2 under no_grad; R1/R2
#     penalty computed on the SAM2-FEATURE manifold (paper-design
#     "Option A"). DDP wraps ONLY heads_module (frozen encoder
#     un-wrapped — find_unused_parameters=False). Critic uses its
#     own DDP wrap; both fire all-reduce on their own params only.
#
# Online learning teacher (back from v8 OFF):
#
#   * real_teacher_train_online=true (online LoRA flow training vs GT).
#   * aux_teacher_loss_weight=1.0 (was 0 in v8 to save memory).
#   * Memory budget OK because distilled critic freed the V+SAM2
#     gen-graph cost.
#
# Inherited from v8:
#   * flash_dmd_paper_aligned_adv=true (dual_grad rollout — paper §3.3
#     Eq. 8-9 with K/V decoupling).
#   * gan_loss_weight=1.0, gan_warmup_steps=1000.
#   * Flash-DMD timestep gating (split_timestep=500), fake_score EMA.
#   * dmd_context=mix, dmd_context_mix_p=0.15.
#   * causal_mask=false (bidir), boundary_vae_roundtrip=true.
#   * dfake_gen_update_ratio=1.
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
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_v9_distilled_lt
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
echo "Starting Phase-1 Action-Forcing DMD + R3GAN v9 (distilled LT critic) on $(date)"
echo "  v9 deltas vs v8: gan_sam2_distilled_critic=true (LatentSAM2Critic),"
echo "                    gan_sam2_pad_to_square=true (480x832 -> 832x832 -> 512),"
echo "                    real_teacher_train_online=true + aux_teacher_loss_weight=1.0,"
echo "                    real_score/fake_score gradient_checkpointing=true,"
echo "                    Sobolev OFF (grad_loss_weight=0) -> critic trained INLINE"
echo "                    so gen consumes a freshly-updated critic each iter."

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
    real_guidance_scale=2.0 \
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
    gan_warmup_steps=1000 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=512 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=500 \
    gan_critic_grad_frames=1 \
    gan_critic_grad_full_every=0 \
    gan_critic_grad_loss_weight=0.0 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=action_forcing_phase1_gan_j${SLURM_JOB_ID}_v9_distilled_lt

echo "Training completed on $(date)"

