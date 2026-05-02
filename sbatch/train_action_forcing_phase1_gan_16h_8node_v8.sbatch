#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-gan-v8-sam2-pixel
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
# Phase-1 Action-Forcing DMD + R3GAN — v8 (SAM2-pixel GAN, paper-scale λ).
#
# Paper-aligned pixel-space GAN: frozen SAM2 Hiera-B+ encoder +
# multi-scale trainable heads on VAE-decoded pixel video. Replaces
# v7's latent 3D-conv disc which was undersized (~4.6M params) and
# missed the structural failure modes (road edges, vehicle outlines,
# lane markings) that drive the visible blur in our outputs.
#
# Recipe vs v7 (Flash-DMD priorities 1+2):
#
#   1. gan_backbone: latent_3d_conv → sam2_pixel
#      Frozen SAM2 image encoder (Hiera-B+) + trainable heads on the
#      FPN; per-frame application + temporal mean pool. Discriminator
#      sees pixel video (decoded via WAN VAE), not latent video — so
#      the structural cues SAM was pretrained for actually drive the
#      adversarial signal.
#
#   1b. flash_dmd_paper_aligned_adv=true (paper §3.3 Eq. 8-9).
#      Two-grad-point rolling rollout: DMD takes grad at a RANDOM
#      exit rung sampled from [0, N-2] (excludes the last rung); the
#      LAST rung's forward is grad-active and produces the gen's
#      "texture refinement" output (= denoising_step_list[-1]'s t,
#      e.g. 178). Adv loss flows ONLY through the last-rung forward;
#      DMD's high-noise grad and the GAN's low-noise grad live in
#      ONE rollout and a single backward populates both via
#      L = L_DMD + λ · L_adv.
#      Cross-block K/V decoupling (paper §3.3): after the last-rung
#      grad forward, the standard no_grad context_noise commit
#      forward overwrites the K/V slots with graph-free tensors —
#      so the next block's exit-flag (DMD-grad) forward reads
#      detached K/V and DMD's grad does NOT chain back through
#      prior blocks' last-rung (low-noise) gen forwards.
#      With this knob on, the per-iter Flash-DMD alternation is
#      bypassed: BOTH DMD and adv fire every iter, summed as
#      L = L_DMD + λ · L_adv.
#
#   2. gan_loss_weight: 0.05 → 1.0 (paper-scale λ on G's adv loss).
#      The latent-disc 0.05 was ~5% of DMD magnitude; pixel-space
#      logits will scale differently. Watch
#      ``train/r3gan_g_loss_weighted`` vs ``generator_dmd_loss`` on
#      the smoke — should be within one OOM at λ=1.0.
#
#   3. gan_warmup_steps: 500 → 1000 (longer ramp at the higher
#      final weight; same per-step weight delta as v7).
#
#   4. real_teacher_train_online: true → false (frozen merged-v14
#      teacher, no online LoRA training).
#   5. real_guidance_scale: 1.0 → 2.0 (compromise between v6/v7's
#      1.0 and the original 3.0).
#   6. aux_teacher_loss_weight: 1.0 → 0.1 (reduced; with online
#      training off, this knob's leverage is mostly gone anyway —
#      the post-critic teacher step also won't fire).
#
# Inherited from v7 unchanged:
#   * Flash-DMD priorities 1 + 2 (timestep-aware DMD/GAN gating +
#     EMA fake_score from generator).
#   * dfake_gen_update_ratio=1, fake_score_ema_weight=0.95.
#   * causal_mask=false (bidir), vae_roundtrip=true,
#     dmd_context_mix_p=0.15.
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
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_v8
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
echo "Starting Phase-1 Action-Forcing DMD + R3GAN v8 (SAM2-pixel) on $(date)"
echo "  v8 deltas vs v7: gan_backbone=sam2_pixel, gan_loss_weight 0.05->1.0,"
echo "                    real_teacher_train_online=false, CFG 1.0->2.0,"
echo "                    aux_teacher_loss_weight 1.0->0.1, gan_warmup 500->1000."

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
    real_teacher_train_online=false \
    real_teacher_causal_mask=false \
    real_guidance_scale=2.0 \
    real_teacher_lr=5e-5 \
    real_teacher_warmup_steps=200 \
    real_teacher_input_source=mix \
    real_teacher_input_mix_gt_p=0.5 \
    dmd_frozen_teacher_pass_enabled=false \
    aux_teacher_loss_weight=0.0 \
    flash_dmd_split_timestep=500 \
    fake_score_ema_weight=0.95 \
    flash_dmd_paper_aligned_adv=true \
    gan_loss_weight=1.0 \
    gan_warmup_steps=1000 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=512 \
    gan_sam2_preserve_aspect=true \
    run_name=action_forcing_phase1_gan_j${SLURM_JOB_ID}_v8_sam2_pixel

echo "Training completed on $(date)"
