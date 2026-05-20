#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-NR-G-v6
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# =====================================================================
# Phase-3 online_teacher_NR_G_v6: three-stage schedule designed to
# bring each loss online only after the prior one has stabilised.
#
# STAGE-BY-STAGE DESIGN
# ---------------------
# v5 collapsed under simultaneous DMD + MANIQA + GAN onset because
# the gen, the aux LoRA, the MANIQA-approx model, and the GAN
# discriminator were all ramping in overlapping windows. v6 stages
# them:
#
#   Stage 1 (steps 0-15):
#     * MANIQA-approx ramps in linearly (perceptual_approx_warmup=0,
#       perceptual_approx_ramp_steps=15). The approx itself is also
#       distilling against real MANIQA — its dense_loss drops fastest
#       through this window. Student-side signal is MANIQA only;
#       DMD off, GAN off.
#     * Online real_score (LoRA aux) ALSO trains from step 0
#       (aux_teacher_start_step=0) so by the time DMD turns on, the
#       LoRA has had a head-start at being a real-score for p_real.
#
#   Stage 2 (steps 20-60):
#     * DMD engages at step 20 (dmd_loss_start_step=20) and ramps to
#       its plateau 0.5 over 150 steps. Effective DMD weight at step
#       60 is 0.5 × 40/150 = 0.133 — non-trivial but well below the
#       collapse threshold seen in v5.
#     * MANIQA is at plateau through this window; the aux LoRA is
#       mid-ramp on its own 80-step warmup (effective aux loss
#       weight at step 20 = 0.25, at step 60 = 0.75, full at step 80).
#     * GAN still off.
#
#   Stage 3 (steps 60-end):
#     * GAN engages — D and gen-side both gate at step 60 simultane-
#       ously (no D head-start, so the critic's gradient field is
#       weak at gate-open and the student doesn't get a sudden
#       directional kick).
#     * Gen-side weight ramps via a QUADRATIC shape over 75 steps
#       (gan_warmup_steps=75, gan_warmup_shape=quadratic). At step
#       61 the effective weight is (1/75)^2 × 0.1 ≈ 1.8e-5 — two
#       orders of magnitude softer than linear at gate-open. Plateau
#       (0.1) at step 135.
#
# WHY THIS ORDERING
# -----------------
# Earlier v3/v4/v5 runs revealed three coupled failure modes:
#
#   (a) GAN onset at gate-open produced fish-scale artifacts because
#       even small linear ramp weights × non-trivial critic gradient
#       fields gave the student a sudden high-frequency directional
#       kick. The quadratic ramp + zero D head-start in v6 reduces
#       the first non-zero gradient by ~100x.
#
#   (b) DMD-MANIQA contention: with DMD ramping in while the MANIQA-
#       approx model was still distilling (v5 step ~30-41), the
#       student saw conflicting signals from a not-yet-converged
#       supervisor and a still-imperfect online LoRA. v6 lets MANIQA
#       fully converge first.
#
#   (c) Online LoRA bootstrap: in v5 the LoRA started training at
#       step 30 alongside DMD turn-on. v6 starts LoRA training at
#       step 0 so 20 steps of pure GT score-matching land before DMD
#       begins pulling on the student through the LoRA.
#
# WHAT INFORMS THE STUDENT vs WHAT INFORMS THE LoRA
# -------------------------------------------------
# Two independent gradient flows. The student is updated by the gen
# optim step driven by MANIQA + DMD + GAN (each with its own ramp).
# The LoRA is updated by its own aux optim step driven by flow-loss
# against GT (aux_teacher_send_student_grad=false severs the LoRA
# aux pass from the student). EMA on the LoRA params (weight 0.99)
# means DMD scoring reads a slow-moving target while the live LoRA
# tracks the latest student outputs.
#
# Key knobs and their effective values:
#
#   knob                                 value     comment
#   ---------------------------------    -------   ------------------
#   max_steps                            250
#   perceptual_approx_warmup_steps       0
#   perceptual_approx_ramp_steps         15        MANIQA full at 15
#   aux_teacher_start_step               0         LoRA trains from 0
#   aux_teacher_loss_warmup_steps        80        LoRA full at 80
#   real_score_ema_weight                0.99      slow EMA target
#   real_teacher_input_source            gt        proper score match
#   dmd_context                          self      no GT mix in DMD
#   dmd_loss_weight                      0.5
#   dmd_loss_start_step                  20
#   dmd_loss_warmup_steps                150       DMD full at 170
#   gan_disc_start_step                  60        D starts at 60
#   gan_critic_warmup_steps              60        gen-side gate at 60
#   gan_warmup_steps                     75        gen-side full at 135
#   gan_warmup_shape                     quadratic 100x softer onset
#   gan_loss_weight                      0.1
#   gan_critic_dense_distillation        true
#   gan_updates_per_step                 1
#   gan_r1_gamma / gan_r2_gamma          2.0 / 2.0
#
# Effective head-weight timeline:
#
#   step   MANIQA   DMD       GAN gen   aux LoRA
#   ------ -------- --------- --------- ---------
#     0    0.000    0.000     0         0.000
#    15    1.000    0.000     0         0.188
#    20    1.000    0.000     0         0.250
#    60    1.000    0.133     0.000     0.750
#    61    1.000    0.137     1.8e-5    0.763
#    75    1.000    0.183     0.004     0.938
#    80    1.000    0.200     0.011     1.000
#   100    1.000    0.267     0.071     1.000
#   135    1.000    0.383     0.100     1.000
#   170    1.000    0.500     0.100     1.000
#   250    1.000    0.500     0.100     1.000
#
# NO-STREAMING MECHANISM (inherited)
# ----------------------------------
# max_rolls_per_ride = 1. Each training step starts a fresh ride
# (iter 1) and resets the streaming state immediately after the
# single chunk is processed.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_NR_G_v6
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8

echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "GAN_SAM2_CKPT=$GAN_SAM2_CKPT"
echo "GAN_SAM2_CFG=$GAN_SAM2_CFG"
echo "Starting Phase-3 online-teacher NR_G v6 (three-stage schedule) on $(date)"
echo "  * 21-frame iter-1-only student rollout, 250-step max."
echo "  * max_rolls_per_ride=1 — streaming resets every step (no iter k>=2)."
echo "  * Teacher: fresh rank-32 LoRA on v14-merged base, EMA(0.99) target."
echo "  * Stage 1 [0, 20): MANIQA ramps in (full at step 15). LoRA"
echo "    aux trains from step 0 (warmup_steps=80, full at step 80)."
echo "    DMD off, GAN off."
echo "  * Stage 2 [20, 60): DMD engages (start=20, warmup=150, cap=0.5)."
echo "    MANIQA at plateau; LoRA still ramping. GAN still off."
echo "  * Stage 3 [60, end): GAN engages with QUADRATIC ramp"
echo "    (gan_warmup_steps=75, shape=quadratic) — first non-zero step"
echo "    delivers ~1.8e-5 vs ~1.3e-3 linear (100x softer onset to"
echo "    suppress fish-scale artifacts at gate-open). D-side starts"
echo "    simultaneously (no head-start) so critic gradients are weak"
echo "    when gen-side opens. GAN full at step 135 (plateau 0.1)."
echo "  * Aux LoRA: aux_teacher_send_student_grad=false (LoRA aux is"
echo "    severed from the student; DMD is the sole student-grad path"
echo "    through the teacher). real_teacher_input_source=gt (proper"
echo "    score-matching, not refinement). real_score_ema_weight=0.99."
echo "  * wandb_project=longlive-phase-3."

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
    wandb_project=longlive-phase-3 \
    max_steps=250 \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    max_rolls_per_ride=1 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=21 \
    num_workers=0 \
    boundary_vae_roundtrip=true \
    local_attn_size_schedule=[[0,21]] \
    dmd_context=self \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=1.0 \
    real_teacher_lr=5e-5 \
    real_teacher_warmup_steps=0 \
    real_teacher_input_source=gt \
    teacher_cadence=fake \
    aux_teacher_p_schedule_enabled=false \
    aux_teacher_send_student_grad=false \
    aux_teacher_start_step=0 \
    aux_teacher_loss_weight=1.0 \
    aux_teacher_loss_warmup_steps=80 \
    aux_teacher_lora_rank=32 \
    aux_teacher_lora_alpha=64 \
    dmd_loss_weight=0.5 \
    dmd_loss_start_step=20 \
    dmd_loss_warmup_steps=150 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    real_score_ema_weight=0.99 \
    gan_loss_weight=0.1 \
    gan_lr=5e-5 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=768 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=60 \
    gan_disc_start_step=60 \
    gan_warmup_steps=75 \
    gan_warmup_shape=quadratic \
    perceptual_approx_warmup_steps=0 \
    perceptual_approx_ramp_steps=15 \
    gan_updates_per_step=1 \
    gan_r1_gamma=2.0 \
    gan_r2_gamma=2.0 \
    gan_sam2_frame_pool=none \
    gan_sam2_encoder_chunk_size=84 \
    maniqa_approx_loss_weight=1.0 \
    maniqa_n_frames=84 \
    maniqa_n_patches_per_frame=8 \
    maniqa_metric_name=maniqa-pipal \
    gan_critic_dense_distillation=true \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=action_forcing_phase3_online_teacher_NR_G_v6_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
