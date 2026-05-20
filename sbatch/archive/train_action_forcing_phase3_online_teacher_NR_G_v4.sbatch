#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-NR-G-v4
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
# Phase-3 online_teacher_NR_G_v3: NR_v2's MANIQA-only base + GAN
# added at step 45 to mitigate DMD-induced collapse around step 61-91.
#
# WHY THIS DESIGN
# ---------------
# NR_v2 (no-GAN, MANIQA-only + DMD from step 55 + aux teacher) was
# stable through step 51 but collapsed in the window [61, 91] once
# DMD's effective weight ramped past ~0.1. Metric trace:
#
#   step  61: gen_loss=0.128 gen_grad=0.27 fake_grad=0.80  (DMD eff=0.12)
#   step  71: gen_loss=0.181 gen_grad=0.66 fake_grad=1.77  (DMD eff=0.32)
#   step  81: gen_loss=0.626 gen_grad=1.01 fake_grad=4.16  (DMD eff=0.52)
#   step  91: gen_loss=0.978 gen_grad=3.42 fake_grad=4.34  (DMD eff=0.72)
#
# dmd_grad itself stayed in [0.45, 0.93] throughout — the DMD signal
# magnitude wasn't exploding, the student's RESPONSE to it was. This
# is classic DMD mode-seeking compounding: as DMD weight ramps up,
# the student locks onto the teacher's modes, the teacher (online
# LoRA) is being trained against student outputs in parallel, and
# the feedback loop closes faster than aux teacher's GT-anchoring
# can compensate. MANIQA alone can't prevent it — it has no
# diversity guarantee.
#
# v3 adds R3GAN as the diversity-preserving regulariser specifically
# tuned to engage BEFORE the step-61 collapse window. R3GAN
# (RpGAN + R1 + R2) is engineered as an anti-mode-collapse loss; the
# discriminator penalises low-diversity student outputs, providing
# the structural-diversity pressure that DMD-only + MANIQA-only
# lacks.
#
# GAN-SIDE TIMING TARGET
# ----------------------
#   * DMD activates step 55, ramps to full at step 105.
#   * NR_v2 collapse window: [61, 91].
#   * GAN gen-side engages step 45 (10 steps before DMD), at meaningful
#     weight by step 61, at plateau before DMD reaches half-strength.
#   * D-training engages step 30 (15-step head-start before gen-side)
#     so D has a meaningful decision boundary when gen-side starts
#     pushing against it.
#
# NO-STREAMING MECHANISM (inherited from NR_v2)
# ---------------------------------------------
# max_rolls_per_ride = 1. Each training step starts a fresh ride
# (iter 1) and resets the streaming state immediately after the
# single chunk is processed. Iter 1 forces ``new_frames =
# chunk_size = 21`` (the full 7-block rollout); iter k≥2 never fires.
#
# Training plan (steps 0..200):
#   * Steps [0, 20):    Fresh rank-32 LoRA FROZEN. No aux loss.
#   * Steps [20, 30):   LoRA aux ramps; D-side, GAN gen-side, DMD
#                       all off. Student steered by MANIQA only.
#   * Steps [30, 45):   D-side training engages (D head-start). MANIQA
#                       still the sole student-side signal.
#   * Steps [45, 55):   GAN gen-side begins ramping. MANIQA + GAN
#                       only on the student.
#   * Steps [55, 200):  DMD engages and ramps; aux LoRA also ramping.
#                       MANIQA + GAN both active throughout.
#
# WHAT INFORMS THE STUDENT vs WHAT INFORMS THE LoRA
# -------------------------------------------------
# Two independent gradient flows: the student is updated by the gen
# optim step (driven by MANIQA + GAN + DMD, all weighted), and the
# LoRA is updated by its own aux optim step (driven by flow-loss
# against GT). aux_teacher_send_student_grad=false severs the LoRA
# aux pass from the student.
#
# Differences vs NR_v2:
#
#   1. GAN ENABLED. gan_enabled=true (was false). R3GAN discriminator
#      built; gen-side push active from step 45.
#
#   2. gan_critic_warmup_steps=45 — gen-side gate. Engages 10 steps
#      before DMD turns on so it's at meaningful weight by step 61.
#
#   3. gan_disc_start_step=30 — D-training starts. 15-step head-start
#      before gen-side engages at step 45.
#
#   4. gan_warmup_steps=30, gan_loss_weight=0.1 — fast ramp so GAN
#      reaches plateau at step 75 (45+30), right as DMD is at 0.066
#      effective (with the new DMD weight=0.5 + warmup=150). Cap 0.1
#      is below the 0.40 empirical stability ceiling so D-domination
#      collapse stays off the table.
#
#   5. gan_updates_per_step=1, gan_critic_dense_distillation=false —
#      D trains at R3GAN paper default 1:1 cadence (not v7/v8's 2:1/
#      5:1 which caused stripe collapse). dense_distillation OFF
#      means the discriminator emits ONE global logit per chunk
#      instead of per-token logits — closes off the per-spatial-token
#      attractor mechanism that produced fish-scales/stripes in
#      v6/v7/v8/v9. Trade-off: GAN provides global "is this real?"
#      signal rather than per-spatial-token supervision; compensated
#      by gan_loss_weight=0.1 (2× v7's 0.05).
#
#   6. DMD softened. dmd_loss_weight=0.5 (was default 1.0) and
#      dmd_loss_warmup_steps=150 (was 50). DMD's effective weight at
#      step 91 (NR_v2's collapse peak) is now 0.12 instead of 0.72
#      — 6× weaker. Reasoning: NR_v2 showed DMD at full strength is
#      overwhelming poor MANIQA; halving the cap and tripling the
#      ramp duration gives the other heads (MANIQA + GAN + aux LoRA)
#      time to build their counter-signal before DMD's compounding
#      pressure becomes dominant.
#
# Effective head-weight schedule under v3:
#
#   step   DMD      GAN gen-side    relative
#   ------ -------- --------------- ---------------------
#   55     0.000    0.033 (45+10)   GAN-only (DMD just on, ramp~0)
#   61     0.020    0.053           GAN 2.7× DMD (good)
#   71     0.053    0.087           GAN 1.6× DMD (still good)
#   75     0.066    0.100 (full)    GAN at plateau, 1.5× DMD
#   91     0.120    0.100           crossover around here; DMD edges out
#   105    0.167    0.100           DMD 1.7× GAN (DMD takes lead)
#   200    0.483    0.100           DMD 4.8× GAN (full-strength DMD reign)
#
# GAN dominates through the previously-collapsing window [61, 91],
# DMD takes the wheel later when the system should be more stable.
#
# Other inherited NR_v2 settings: aux_teacher_loss_weight=1.0,
# aux_teacher_loss_warmup_steps=180 (slow ramp, full at step 200).
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase1_gan.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_NR_G_v4
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
echo "Starting Phase-3 online-teacher NR_G v3 (no streaming, GAN enabled at step 45) on $(date)"
echo "  * 21-frame iter-1-only student rollout, 200-step max."
echo "  * max_rolls_per_ride=1 — streaming resets every step (no iter k>=2)."
echo "  * Teacher: fresh rank-32 LoRA on v14-merged base."
echo "  * Steps [0, 20): LoRA frozen. Steps [20, 200): LoRA aux ramps 0 -> 1.0."
echo "  * Steps [0, 55): MANIQA only on student (DMD off). GAN gen-side"
echo "    engages step 45 (10 steps before DMD)."
echo "  * D-training (gan_disc_start_step=30): 15-step head-start before"
echo "    gen-side engages at step 45."
echo "  * DMD softened: weight=0.5, start=55, warmup=150 (full at step 205,"
echo "    effective 0.483 at step 200 — gentler than NR_v2's 1.0 cap)."
echo "  * GAN cap=0.1, gan_warmup_steps=30 (full at step 75), dense_distill=off,"
echo "    D-cadence=1:1 — mitigates DMD-induced mode collapse seen in NR_v2"
echo "    step 61-91 window."
echo "  * aux_teacher_send_student_grad=false (DMD is sole student-grad path"
echo "    from the teacher; aux LoRA optim is severed from the student)."
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
    max_steps=200 \
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
    aux_teacher_start_step=30 \
    aux_teacher_loss_weight=1.0 \
    aux_teacher_loss_warmup_steps=180 \
    aux_teacher_lora_rank=32 \
    aux_teacher_lora_alpha=64 \
    dmd_loss_weight=0.5 \
    dmd_loss_start_step=55 \
    dmd_loss_warmup_steps=150 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    real_score_ema_weight=0.95 \
    gan_loss_weight=0.1 \
    gan_lr=5e-5 \
    gan_backbone=sam2_pixel \
    gan_sam2_checkpoint_path=$GAN_SAM2_CKPT \
    gan_sam2_config_path=$GAN_SAM2_CFG \
    gan_sam2_resolution=768 \
    gan_sam2_preserve_aspect=false \
    gan_sam2_pad_to_square=true \
    gan_sam2_distilled_critic=true \
    gan_critic_warmup_steps=30 \
    gan_disc_start_step=20 \
    gan_warmup_steps=30 \
    gan_updates_per_step=1 \
    gan_r1_gamma=5.0 \
    gan_r2_gamma=5.0 \
    gan_sam2_frame_pool=none \
    gan_sam2_encoder_chunk_size=42 \
    maniqa_approx_loss_weight=1.0 \
    maniqa_n_frames=84 \
    maniqa_n_patches_per_frame=8 \
    maniqa_metric_name=maniqa-pipal \
    gan_critic_dense_distillation=true \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=action_forcing_phase3_online_teacher_NR_G_v4_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
