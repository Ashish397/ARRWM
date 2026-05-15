#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-NR-G-v12
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
# Phase-3 online_teacher_NR_G_v12: v10 + AR-NOISE TEACHER AUGMENTATION.
#
# v11 NEW IDEA — AR-NOISE-AWARE TEACHER TRAINING
# ----------------------------------------------
# All prior runs trained the teacher (LoRA aux pass) on
# add_noise(GT, ε, t) with ε ~ N(0, I). At deployment, the teacher
# scores noised STUDENT outputs, and the student's outputs contain
# autoregressive (AR) noise — a structured non-Gaussian distortion
# that's WHAT THE TEACHER NEEDS TO REMOVE. Training on pure Gaussian
# leaves a train/deploy distribution mismatch.
#
# v11 closes this gap. On every gen step the DMD scoring computes
# pred_real = teacher(noised_student_chunk). The difference
# (chunk - pred_real) is the teacher's empirical estimate of the
# AR-noise direction in the student's output. We stash this and use
# it to perturb the teacher's training noise:
#
#     noise_for_teacher = ε + scale * (chunk - pred_real).detach()
#     noisy_input        = (1-t)*GT + t*noise_for_teacher
#     loss               = (flow_pred - (noise_for_teacher - GT))²
#
# The target stays GT (preserves score-matching contract — student's
# creative latitude is not pinched), and the score-matching identity
# is approximately preserved because ε remains the dominant component.
# The teacher learns to denoise inputs that contain BOTH random
# Gaussian noise AND a structured AR-noise component — exactly what
# it'll see at deployment.
#
# Mode "two_pass" (v11 default): each iter runs TWO teacher forwards:
#   pass 1 — pure ε (preserves the score-matching identity exactly
#            for ∇log p_real(x_t), keeping DMD math unbiased)
#   pass 2 — ε + scale*ar_noise (teaches AR-denoising)
# Losses summed. Doubles teacher forward+backward cost (~30% wallclock
# per iter on the LoRA path with teacher_cadence='fake' already
# running 5 cycles).
# Switch to "combined" mode (single forward with combined noise) to
# halve cost at the price of slightly biasing the score-matching
# identity. "off" reverts to v10 behaviour.
#
# scale=1.0 default: AR noise added at its raw magnitude. Track
# ``aux_teacher_ar_noise_rms`` in wandb to see the typical magnitude
# and tune scale accordingly.
#
# v10 LEGACY — DOUBLED GAN, GT-LATENT ANCHOR (kept for v11)
# ---------------------------------------------------------
# v10 changes inherited unchanged: gan_loss_weight=1.0 (5x v9,
# 10x v7), gt_latent_mse_loss_weight=0.1, gt_latent_mae_loss_weight=
# 0.05. See v10 sbatch header for the full rationale.
#
# v9 POST-MORTEM — DOUBLED GAN PREVENTED COLLAPSE, RAN 2H FULL TERM
# ----------------------------------------------------------------
# v9 (gan_loss_weight=0.2) was a clear improvement over v7. It hit
# the 2h SLURM walltime at step ~210/250 without ever showing the
# hard mode-seeking collapse signature. dmd_grad held in 0.85-0.99
# the whole run (vs v7's drop to 0.66 at step 151, v6's 0.55 at
# step 81). gen_loss oscillated between -0.21 and +0.41 with
# consistent recovery (step 181 surged back to +0.343 like step 51).
#   step 51:  gen_loss +0.412, dmd_grad 0.91   (v6/v7 surge mirror)
#   step 81:  gen_loss -0.207, dmd_grad 0.90   (v6 collapse moment;
#                                               v9 dmd_grad stayed
#                                               healthy)
#   step 121: gen_loss -0.010, dmd_grad 0.96   (v7 gray-collapse
#                                               moment; v9 stable)
#   step 151: gen_loss +0.029, dmd_grad 0.98   (v7 hard collapse;
#                                               v9 clean recovery)
#   step 181: gen_loss +0.343, dmd_grad 0.92   (strong bounce-back)
#   step 201: gen_loss +0.098, dmd_grad 0.72   (briefly elevated
#                                               DMD pull but no
#                                               collapse signature)
# c_corr still oscillated between +0.97 and -0.99 (disc within-batch
# anti-alignment persisted) but the doubled GAN provided enough
# amplitude during aligned phases to keep the student stable through
# the anti-aligned phases. d_real-d_fake separation kept growing
# from 0.000 (step 61) to 0.16 (step 201) — disc IS learning on
# average even if per-sample ranking flips intermittently.
#
# v10 STRATEGY — PUSH GAN HARDER + ADD GT-LATENT ANCHOR
# -----------------------------------------------------
# Two changes vs v9:
#   (a) Bump gan_loss_weight 0.2 -> 1.0. This flips the head balance:
#       DMD:GAN plateau ratio was 2.5:1 in v9, becomes 1:2 in v10 (GAN
#       now dominates DMD by 2x at plateau). GAN is now the primary
#       student trainer; DMD becomes a secondary regulariser.
#   (b) Add constant-weight latent-space recon losses against GT:
#         gt_latent_mse_loss_weight = 0.1
#         gt_latent_mae_loss_weight = 0.05
#       Both fire every step from step 0 with no ramp or warmup —
#       a small direct anti-drift anchor pulling pred_image toward
#       the GT latent window. Independent of DMD/GAN/MANIQA chains
#       (no critic, no approx, no VAE — pure latent-space L1+L2).
#       Implemented in trainer ``_compute_gt_latent_recon_loss``.
# Accepted risk: during disc-anti-aligned phases the 5x stronger GAN
# pushes 5x harder in the wrong direction. v9 absorbed that shock
# because GAN was still below DMD; with GAN now > DMD, an extended
# anti-aligned phase could pull the student off-manifold faster than
# DMD can correct. If v10 collapses, the disc itself is confirmed
# as the bottleneck and the next lever is disc-side tuning
# (R1/R2 gamma up, longer disc head-start, lower disc LR) — NOT
# pushing GAN weight higher.
#
# Alternatively, if v10 holds without collapse, that's evidence the
# disc anti-alignment is a tolerable transient state and the system
# is GAN-bound rather than disc-bound at higher GAN weights.
#
# STAGE-BY-STAGE DESIGN (inherited from v7)
# -----------------------------------------
#
#   Stage 1 (steps 0-15):
#     * MANIQA-approx ramps in linearly (perceptual_approx_warmup=0,
#       perceptual_approx_ramp_steps=15). The approx itself is also
#       distilling against real MANIQA — its dense_loss drops fastest
#       through this window. Student-side signal is MANIQA only;
#       DMD off, GAN off.
#     * Online real_score (LoRA aux) trains from step 0
#       (aux_teacher_start_step=0).
#
#   Stage 2 (steps 20-60):
#     * DMD engages at step 20 (dmd_loss_start_step=20) and ramps to
#       its plateau 0.5 over 150 steps.
#     * MANIQA at plateau; LoRA still ramping (full at step 80).
#     * GAN still off.
#
#   Stage 3 (steps 60-end):
#     * GAN engages — D and gen-side both gate at step 60 simultane-
#       ously (no D head-start).
#     * Gen-side weight ramps via a QUADRATIC shape over 25 steps
#       (gan_warmup_steps=25, gan_warmup_shape=quadratic). Plateau
#       (1.0) at step 85.
#
# WHAT INFORMS THE STUDENT vs WHAT INFORMS THE LoRA
# -------------------------------------------------
# Two independent gradient flows. The student is updated by the gen
# optim step driven by MANIQA + DMD + GAN (each with its own ramp).
# The LoRA is updated by its own aux optim step driven by flow-loss
# against GT (aux_teacher_send_student_grad=false severs the LoRA
# aux pass from the student). v8 confirmed that flipping this to
# true causes deeper collapse — keep false. EMA on the LoRA params
# (weight 0.99) means DMD scoring reads a slow-moving target while
# the live LoRA tracks the latest student outputs.
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
#   aux_teacher_send_student_grad        false     (v8 collapsed when true)
#   aux_teacher_ar_noise_mode            two_pass  same as v11
#   aux_teacher_ar_noise_scale           0.5       v12: half raw AR magnitude (was 1.0)
#   real_score_ema_weight                0.99      slow EMA target
#   real_teacher_input_source            gt        proper score match
#   dmd_context                          self      no GT mix in DMD
#   dmd_loss_weight                      0.5
#   dmd_loss_start_step                  20
#   dmd_loss_warmup_steps                150       DMD full at 170
#   gan_disc_start_step                  60        D starts at 60
#   gan_critic_warmup_steps              60        gen-side gate at 60
#   gan_warmup_steps                     25        gen-side full at 85
#   gan_warmup_shape                     quadratic gentle but quick
#   gan_loss_weight                      1.0       <-- v10 change (5x v9, 10x v7)
#   gan_critic_dense_distillation        true
#   gan_updates_per_step                 1
#   gan_r1_gamma / gan_r2_gamma          2.0 / 2.0
#
# Effective head-weight timeline (GAN plateau is 1.0 vs v9's 0.2):
#
#   step   MANIQA   DMD       GAN gen   aux LoRA
#   ------ -------- --------- --------- ---------
#     0    0.000    0.000     0         0.000
#    15    1.000    0.000     0         0.188
#    20    1.000    0.000     0         0.250
#    60    1.000    0.133     0.000     0.750
#    61    1.000    0.137     1.6e-3    0.763
#    65    1.000    0.150     0.0400    0.813
#    70    1.000    0.167     0.1600    0.875
#    75    1.000    0.183     0.3600    0.938
#    80    1.000    0.200     0.6400    1.000
#    85    1.000    0.217     1.0000    1.000
#   135    1.000    0.383     1.0000    1.000
#   170    1.000    0.500     1.0000    1.000
#   250    1.000    0.500     1.0000    1.000
#
# DMD:GAN ratio at plateau is now 0.5:1.0 = 0.5 — GAN dominates DMD
# by 2x (v9 was 2.5, v7 was 5.0). This is a major rebalancing — the
# student is now primarily GAN-trained.
#
# NO-STREAMING MECHANISM (inherited)
# ----------------------------------
# max_rolls_per_ride = 1.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_NR_G_v12
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
echo "Starting Phase-3 online-teacher NR_G v11 (AR-noise teacher aug + v10 base) on $(date)"
echo "  * 21-frame iter-1-only student rollout, 250-step max."
echo "  * max_rolls_per_ride=1 — streaming resets every step (no iter k>=2)."
echo "  * Teacher: fresh rank-32 LoRA on v14-merged base, EMA(0.99) target."
echo "  * v9 POST-MORTEM: v9 (gan_loss_weight=0.2) ran the full 2h walltime"
echo "    to step ~210 without hard collapse. dmd_grad held 0.85-0.99 throughout"
echo "    (vs v7's drop to 0.66 at step 151). v10 pushes gan_loss_weight 0.2 -> 1.0"
echo "    (DMD:GAN plateau ratio flips from 2.5:1 to 1:2, GAN now dominates DMD)."
echo "  * v10 ALSO adds constant-weight latent recon: gt_latent_mse=0.1,"
echo "    gt_latent_mae=0.05 — direct anti-drift anchor between pred_image and GT,"
echo "    fires every step (no ramp). Independent of DMD/GAN/MANIQA pipelines."
echo "  * v8 POST-MORTEM: aux_teacher_send_student_grad=true caused deeper"
echo "    collapse (gen_loss -0.36 at step 101 vs v7's +0.15) — kept false here."
echo "  * Stage 1 [0, 20): MANIQA ramps in (full at step 15). LoRA"
echo "    aux trains from step 0 (warmup_steps=80, full at step 80)."
echo "    DMD off, GAN off."
echo "  * Stage 2 [20, 60): DMD engages (start=20, warmup=150, cap=0.5)."
echo "    MANIQA at plateau; LoRA still ramping. GAN still off."
echo "  * Stage 3 [60, end): GAN engages with QUADRATIC ramp"
echo "    (gan_warmup_steps=25, shape=quadratic). D-side starts simul-"
echo "    taneously (no head-start). GAN full at step 85 (plateau 1.0)."
echo "  * v11 NEW: aux_teacher_ar_noise_mode=two_pass (scale=1.0)."
echo "    Teacher trains on (pure Gaussian) AND (Gaussian + AR-noise"
echo "    estimate from chunk - pred_real) — closes the train/deploy"
echo "    distribution mismatch on the teacher's noise input. Doubles"
echo "    teacher fwd+bwd cost; track aux_teacher_ar_active /"
echo "    aux_teacher_ar_noise_rms in wandb."
echo "  * Aux LoRA: aux_teacher_send_student_grad=false (LoRA aux is"
echo "    severed from the student; DMD is the sole student-grad path"
echo "    through the teacher). real_teacher_input_source=gt."
echo "    real_score_ema_weight=0.99."
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
    sample_fps=20 \
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
    aux_teacher_ar_noise_mode=two_pass \
    aux_teacher_ar_noise_scale=0.5 \
    dmd_loss_weight=0.5 \
    dmd_loss_start_step=60 \
    dmd_loss_warmup_steps=150 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    real_score_ema_weight=0.99 \
    gan_loss_weight=0.5 \
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
    gan_warmup_steps=25 \
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
    gt_latent_mse_loss_weight=0.1 \
    gt_latent_mae_loss_weight=0.05 \
    run_name=action_forcing_phase3_online_teacher_NR_G_v12_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
