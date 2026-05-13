#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-NR-G-v9
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
# Phase-3 online_teacher_NR_G_v9: identical to v7 except
# gan_loss_weight = 0.2 (was 0.1 in v7) — GAN doubled.
#
# v7 POST-MORTEM — WENT FURTHER, STILL GRAY-COLLAPSED
# ---------------------------------------------------
# v7 was a big improvement over v6: gan_warmup_steps=25 (quadratic)
# let the GAN reach full strength (0.1) at step 85 instead of step
# 135. The student recovered from the step-81 DMD surge:
#   step 51:  gen_loss +0.371  (DMD surge, GAN off)
#   step 71:  gen_loss -0.049  (GAN ramp at 0.0019 — weak)
#   step 81:  gen_loss -0.155  (GAN ramp at 0.044 — still weak)
#   step 91:  gen_loss +0.075  (RECOVERED, GAN approaches 0.1)
#   step 101: gen_loss +0.154  (best point, full GAN + aligned disc)
# But by step 111 the discriminator flipped to anti-aligned
# (c_corr = -0.99 despite d_real-d_fake mean still correct) and the
# student fell back:
#   step 111: gen_loss -0.121, c_corr -0.99   (disc within-batch
#                                              ranks real BELOW fake)
#   step 121: gen_loss -0.096                  (GRAY COLLAPSE
#                                              observed in video)
#   step 131: gen_loss +0.013, c_corr +0.46    (transient recovery)
#   step 141: gen_loss +0.074
#   step 151: gen_loss -0.181, dmd_grad 0.66   (mode-seeking
#                                              signature returns)
# The failure mode is NOT pure DMD dominance — it's disc oscillation
# between aligned (c_corr ~+0.9) and anti-aligned (c_corr ~-0.99)
# states. During the aligned phases v7's GAN successfully regularises
# the student; during the anti-aligned phases the GAN gradient pushes
# in the wrong direction and DMD wins.
#
# v9 STRATEGY — STRONGER GAN (DOUBLED)
# ------------------------------------
# Bump gan_loss_weight 0.1 -> 0.2 (doubled). This:
#   * Makes the aligned-disc phases much more anti-collapse (twice
#     the GAN pressure when the disc IS giving the right signal).
#   * Brings DMD:GAN plateau ratio from 5:1 (v7) down to 2.5:1.
# Accepted risk: during disc-anti-aligned phases the doubled GAN
# pushes harder in the wrong direction. The bet is that the aligned
# phases of v7 were close to working (recovered to gen_loss +0.154
# at step 101) and just need more pull; if the anti-aligned signal
# deepens the gray-collapse phase, v10 will need disc tuning
# (R1/R2 gamma up, longer disc head-start, lower disc LR) regardless.
#
# If v9 collapses at the same step pattern with similar visuals
# despite the doubled GAN, the disc oscillation is confirmed as the
# real bottleneck and disc-side tuning is the next lever.
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
#       (0.2) at step 85.
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
#   gan_loss_weight                      0.2       <-- v9 change (doubled)
#   gan_critic_dense_distillation        true
#   gan_updates_per_step                 1
#   gan_r1_gamma / gan_r2_gamma          2.0 / 2.0
#
# Effective head-weight timeline (GAN plateau is 0.2 vs v7's 0.1):
#
#   step   MANIQA   DMD       GAN gen   aux LoRA
#   ------ -------- --------- --------- ---------
#     0    0.000    0.000     0         0.000
#    15    1.000    0.000     0         0.188
#    20    1.000    0.000     0         0.250
#    60    1.000    0.133     0.000     0.750
#    61    1.000    0.137     3.2e-4    0.763
#    65    1.000    0.150     0.0080    0.813
#    70    1.000    0.167     0.0320    0.875
#    75    1.000    0.183     0.0720    0.938
#    80    1.000    0.200     0.1280    1.000
#    85    1.000    0.217     0.2000    1.000
#   135    1.000    0.383     0.2000    1.000
#   170    1.000    0.500     0.2000    1.000
#   250    1.000    0.500     0.2000    1.000
#
# DMD:GAN ratio at plateau is now 0.5:0.2 = 2.5 (v7 was 5.0).
#
# NO-STREAMING MECHANISM (inherited)
# ----------------------------------
# max_rolls_per_ride = 1.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_NR_G_v9
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
echo "Starting Phase-3 online-teacher NR_G v9 (gan_loss_weight=0.2) on $(date)"
echo "  * 21-frame iter-1-only student rollout, 250-step max."
echo "  * max_rolls_per_ride=1 — streaming resets every step (no iter k>=2)."
echo "  * Teacher: fresh rank-32 LoRA on v14-merged base, EMA(0.99) target."
echo "  * v7 POST-MORTEM: v7 reached step 101 with gen_loss +0.154 then"
echo "    fell back to gray-collapse at step 121 (gen_loss ~-0.10) as the"
echo "    discriminator flipped to within-batch anti-alignment (c_corr ~-0.99"
echo "    while d_real-d_fake mean stayed correct). v9 bumps gan_loss_weight"
echo "    0.1 -> 0.2 (DMD:GAN plateau ratio 5:1 -> 2.5:1) to give the GAN"
echo "    more pull during disc-aligned phases."
echo "  * v8 POST-MORTEM: aux_teacher_send_student_grad=true caused deeper"
echo "    collapse (gen_loss -0.36 at step 101 vs v7's +0.15) — kept false here."
echo "  * Stage 1 [0, 20): MANIQA ramps in (full at step 15). LoRA"
echo "    aux trains from step 0 (warmup_steps=80, full at step 80)."
echo "    DMD off, GAN off."
echo "  * Stage 2 [20, 60): DMD engages (start=20, warmup=150, cap=0.5)."
echo "    MANIQA at plateau; LoRA still ramping. GAN still off."
echo "  * Stage 3 [60, end): GAN engages with QUADRATIC ramp"
echo "    (gan_warmup_steps=25, shape=quadratic). D-side starts simul-"
echo "    taneously (no head-start). GAN full at step 85 (plateau 0.2)."
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
    gan_loss_weight=0.2 \
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
    run_name=action_forcing_phase3_online_teacher_NR_G_v9_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
