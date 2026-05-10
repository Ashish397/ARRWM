#!/bin/bash
#SBATCH --job-name=action-forcing-phase1-gan-v25
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
# v25: same as v24 + dense per-token GAN-D distillation.
#
#   gan_critic_dense_distillation=true (NEW knob). The latent
#   SAM2 critic now ALSO trains on the disc's per-token logit map:
#   disc.forward_dense_heads(target_h=8, target_w=13) → pool 4
#   pixel-frames per latent → MSE against critic.forward(dense=True)
#   = [B, F_lat, 8, 13]. ~104x more constraints per sample than
#   the legacy scalar/per-frame MSE. Scalar MSE preserved as an
#   additive complementary signal in L_value. Multi-noise interp /
#   FD-Sobolev / Sobolev paths stay scalar.
#
# v24 (inherited): same as v23 except aux_clean_x_mix_p 1.0 -> 0.15.
# Matches v22's typical LoRA-aux behavior (~15% GT / 85% student in
# clean_x), now STATICALLY held on the aux side and decoupled from
# the DMD scoring path's mix_p (which still steps 0.15 -> 0.7 at
# step 100). The decoupling is the architectural fix; the value
# choice keeps the LoRA training distribution numerically equivalent
# to what it saw in v22 most of the time.
#
# All other v23 changes preserved. v23 deltas vs v22:
#
#   1. MANIQA NR-IQA loss ENABLED at weight=1.0. 4 random 224x224
#      patches per latent frame, all 84 latent frames scored every
#      iter. Sparse spatial target — patches map back to specific
#      approx-grid tokens, rest masked. Gen-side: -quality.mean()
#      (gen wants quality HIGH).
#   2. LPIPS approx + MS-SSIM approx + MSE approx all DISABLED.
#      Replaces them with MANIQA only, per the user's "approach (b)"
#      decision.
#   3. gan_sam2_encoder_chunk_size 21 -> 84 (no chunking).
#      Confirmed -17 GB peak vs chunk_size=21 in smoke 4465578.
#   4. action_critic gen-side z-guidance now routes to
#      paper_aligned_x0_for_adv (the lowest-noise gen forward) when
#      available — Phase A change. Dormant in v23 since action_critic
#      is OFF; matters when re-enabled.
#   5. aux_clean_x_mix_p=0.15 (NEW knob; matches v22 typical
#      behavior). Decouples the LoRA aux pass's clean_x mix from
#      the DMD scoring's dmd_context_mix_p. v22's LoRA aux was
#      effectively seeing ~15% GT / 85% student most of the time
#      (sensor only flipped to 0.7 occasionally). v24 holds that
#      0.15 mix STATICALLY on the aux side, regardless of how the
#      DMD scoring path's mix_p evolves (steps 0.15 -> 0.7 at step
#      100 — see (6)). The two paths are decoupled but the LoRA's
#      training distribution matches v22's typical state.
#   6. Sensor gate DISABLED in v23 (was active in v22). Replaced by a
#      hard step-switch: mix_p = 0.15 for step < 100, 0.7 from
#      step 100 onward. Keeps the LoRA training on a known
#      distribution for the first 100 steps (warmup), then
#      hard-bolsters the DMD scoring's GT context once the LoRA is
#      stable. New trainer knob: dmd_context_mix_p_step_switch_at.
#   7. real_teacher_input_source=blend (NEW mode).
#      DETERMINISTIC linear blend on the LoRA's noisy_input:
#         noise_base = 0.7 * gt_target + 0.3 * student_chunk
#      every iter (vs v22's per-iter Bernoulli coin between fully-GT
#      and fully-student). Lower variance, and the implicit gradient
#      channel through ``chunk`` is ALWAYS live (since p=0.7<1.0)
#      instead of being off 70% of iters. Trade-off: trains on a
#      synthetic distribution that matches NEITHER eval-time query
#      (LoRA is queried on pure-student during DMD scoring).
#
# CFG / LoRA-online / GAN backbone unchanged from v22.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_v25
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
echo "Starting Phase-1 Action-Forcing DMD + R3GAN v25 on $(date)"
echo "v25 from v24: gan_critic_dense_distillation=true — critic"
echo "trains on disc per-token logit map [B, F_lat, 8, 13] (104x more"
echo "constraints/sample) in addition to legacy scalar MSE."
echo "v24 from v23: aux_clean_x_mix_p 1.0 -> 0.15 (matches v22's"
echo "typical LoRA-aux behavior; decoupled from DMD scoring's mix_p)."
echo "v23 from v22:"
echo "  * MANIQA NR-IQA loss ENABLED at weight=1.0 (n_frames=84, n_patches=4)."
echo "  * LPIPS / MS-SSIM / MSE approxes DISABLED (replace by MANIQA)."
echo "  * gan_sam2_encoder_chunk_size 21 -> 84 (memory win)."
echo "  * action_critic gen-side z-guidance routes to paper_aligned_x0."
echo "  * aux_clean_x_mix_p=0.15 — LoRA aux pass holds 15%-GT/85%-self"
echo "    static, decoupled from DMD scoring's mix_p."
echo "  * Sensor DISABLED. Hard step-switch: mix_p 0.15 -> 0.7 at"
echo "    step 100 (dmd_context_mix_p_step_switch_at=100)."
echo "  * real_teacher_input_source=blend (deterministic linear blend"
echo "    0.7*gt + 0.3*student every iter; was Bernoulli mix in v22)."


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
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    max_rolls_per_ride=300 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=21 \
    num_workers=0 \
    boundary_vae_roundtrip=false \
    local_attn_size_schedule=[[0,21]] \
    dmd_context=self \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=1.0 \
    real_teacher_lr=5e-5 \
    real_teacher_warmup_steps=200 \
    real_teacher_input_source=blend \
    real_teacher_input_mix_gt_p=None \
    aux_clean_x_mix_p=0.7 \
    aux_teacher_loss_weight=1.0 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    gan_loss_weight=1.0 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=768 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=10 \
    gan_warmup_steps=80 \
    gan_sam2_frame_pool=none \
    gan_sam2_encoder_chunk_size=84 \
    maniqa_approx_loss_weight=1.0 \
    maniqa_n_frames=84 \
    maniqa_n_patches_per_frame=8 \
    maniqa_metric_name=maniqa-pipal \
    gan_critic_dense_distillation=true \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=action_forcing_phase1_gan_j${SLURM_JOB_ID}_v25

echo "Training completed on $(date)"
