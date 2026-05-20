#!/bin/bash
#SBATCH --job-name=action-forcing-phase3-online-teacher-NR-G-v14
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
# Phase-3 online_teacher_NR_G_v14: v13 BASE + AGGRESSIVE TUNING.
#
# v14 vs v13 — four knob changes (LR + start_step retuning)
# ---------------------------------------------------------
# v13's first 60 steps showed a markedly gentler regime than v9-v12:
#   step 51 gen_loss:  v6 +0.35,  v7 +0.37,  v9 +0.41,
#                      v10 +0.41, v12 +0.58, v13 +0.099 (4-6x lower)
#   step 51 dmd_grad:  v9-v12 ~0.95, v13 0.40 (half)
# This is because max_gradient_chunks=2 (gradient on 6/21 frames) +
# dfake_gen_update_ratio=2 (fake_score chases 2.5x slower) + new
# AR-noise anchor combine to dramatically reduce the per-step
# training pressure. v13 is structurally anti-collapse but at the
# cost of being UNDER-TRAINED within the 2h walltime — projected
# end of run is step ~170/250, DMD at 73% of plateau.
#
# v14 compensates with more aggressive LRs and an earlier DMD start:
#  (1) lr (gen): 1e-5 (YAML default) -> 2e-5
#      Doubled gen LR so the student converges faster despite the
#      smaller effective gradient window.
#  (2) real_teacher_lr: 5e-5 -> 7.5e-5
#      1.5x faster LoRA real_score so it adapts to the AR-augmented
#      input distribution sooner. fake_lr stays at default (1e-5)
#      so dfake_gen_update_ratio=2 remains effective at slowing the
#      fake_score chase relative to the gen.
#  (3) dmd_loss_start_step: 60 -> 40
#      Earlier DMD onset. With v13's anti-collapse mechanisms
#      (AR-noise teacher anchor, GT-recon constants, max_gradient_
#      chunks=2 reducing exposure to noise-dominated late chunks),
#      the system can tolerate DMD signal sooner. Reaches DMD
#      plateau ~30 steps earlier, useful within 2h walltime.
#  (4) gan_loss_weight: 0.2 -> 0.3
#      Modest 1.5x bump on GAN. v13's reduced fake_score chase
#      slightly weakens the DMD signal; a slightly stronger GAN
#      compensates without re-introducing v10's over-GAN regime.
#
# Everything else carries over from v13 unchanged: max_gradient_
# chunks=2, dfake_gen_update_ratio=2, AR-noise combined mode at
# scale=1.0 with α(t)=clamp((1-σ)/σ,10), seed_mask_chunks=1,
# burn_in_steps=20, variance_renorm=true (decay 0.99), CFG off,
# gt_latent_mse=0.1, gt_latent_mae=0.05.
#
# v13 LEGACY DOCS BELOW (kept for context)
# ----------------------------------------
#
# v13 vs v10 — three structural changes
# -------------------------------------
#  (A) CFG off: real_guidance_scale 1.0 -> 0.0. The teacher's
#      conditional forward is now used directly with no CFG
#      extrapolation. Cheaper, simpler, removes the (cond - uncond)
#      contribution from every teacher output.
#
#  (B) gan_loss_weight 0.5 -> 0.2 (lighter GAN). With AR-noise
#      anchoring the teacher into the student's actual distribution,
#      we expect to need less raw GAN pressure to stay on-manifold.
#
#  (C) AR-noise-aware aux teacher training. The teacher is no longer
#      trained only on add_noise(GT, ε, t); per iter the aux pass
#      now also sees a clean-side input perturbed by the empirical
#      AR-drift of the student rollout:
#         ar_residual = (chunk - gt_target).detach()        [x0-space]
#         α(t)        = clamp((1 - σ_t) / σ_t, max=α_max)    [flow conv]
#         ar_norm     = ar_residual / EMA(rms(ar_residual))  [renorm]
#         seed_mask   = linear ramp 0->1 over first K chunks  [drift-aware]
#         burn_in     = min(1, step / N)                      [warmup]
#         noise_base_ar = gt + scale·burn_in·α(t)·seed_mask·ar_norm
#         noisy_input   = add_noise(noise_base_ar, ε, t)
#      Target stays ``ε - gt_target`` (pure GT), so the teacher
#      learns to denoise drift-perturbed inputs back toward
#      unperturbed GT. Mode=combined (single forward), scale=1.0,
#      α_max=10, seed_mask_chunks=1, burn_in_steps=20,
#      variance_renorm=true (EMA decay 0.99).
#
#      Rationale for using (chunk - GT) instead of (chunk - pred_real)
#      from v11/v12: GT is the actual reference at the seed-adjacent
#      chunks; the residual is guaranteed to be AR drift there. As
#      the rollout extends, drift compounds and GT stops being a
#      faithful predictor of the student — but (D) below restricts
#      the gradient window to the seed-adjacent rolls anyway, so the
#      late-rollout-drift concern doesn't bite.
#
#  (D) max_gradient_chunks=2. Only the first two rolled chunks
#      (frames 0..5) get gradient for DMD score, aux teacher, and
#      critic losses. Later chunks are noise-dominated and discarded.
#      Composes with the existing v14-LoRA last-chunk boundary mask
#      (the intersection of "first 2 chunks" and "not last chunk").
#      Wired via the new ``_dmd_score_grad_mask`` cap.
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
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_online_teacher_NR_G_v14
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
echo "Starting Phase-3 online-teacher NR_G v14 (v13 base + aggressive LR/start tuning) on $(date)"
echo "  * v14 changes vs v13: lr 1e-5->2e-5, real_teacher_lr 5e-5->7.5e-5,"
echo "    dmd_loss_start_step 60->40, gan_loss_weight 0.2->0.3."
echo "    All other v13 anti-collapse mechanisms unchanged."
echo "  * 21-frame iter-1-only student rollout, 250-step max."
echo "  * max_rolls_per_ride=1 — streaming resets every step (no iter k>=2)."
echo "  * Teacher: fresh rank-32 LoRA on v14-merged base, EMA(0.99) target."
echo "  * v13 NEW (vs v10):"
echo "      - CFG off (real_guidance_scale=0)."
echo "      - gan_loss_weight 0.5 -> 0.2 (lighter GAN; AR-noise carries weight)."
echo "      - aux_teacher_ar_noise_mode=combined; AR noise = (chunk-GT) fresh"
echo "        each iter, added to GT before noising (CLEAN-side perturbation)."
echo "        Flow-space scaling α(t)=clamp((1-σ)/σ, max=10), seed-mask ramp"
echo "        over first 1 chunk, 20-step burn-in, EMA-renormalised RMS."
echo "      - max_gradient_chunks=2: only first 2 rolled chunks (frames 0..5)"
echo "        carry gradient for DMD score AND aux teacher. Later chunks are"
echo "        noise-dominated and discarded."
echo "  * gt_latent_mse=0.1, gt_latent_mae=0.05 — direct anti-drift latent"
echo "    L1+L2 against GT, every step, no ramp (carried from v10)."
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
    dfake_gen_update_ratio=2 \
    model_kwargs.local_attn_size=21 \
    num_workers=0 \
    sample_fps=20 \
    boundary_vae_roundtrip=true \
    local_attn_size_schedule=[[0,21]] \
    dmd_context=self \
    real_teacher_train_online=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=0.0 \
    lr=2e-5 \
    real_teacher_lr=7.5e-5 \
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
    dmd_loss_start_step=40 \
    dmd_loss_warmup_steps=150 \
    dmd_frozen_teacher_pass_enabled=false \
    fake_score_ema_weight=0.95 \
    real_score_ema_weight=0.99 \
    gan_loss_weight=0.3 \
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
    aux_teacher_ar_noise_mode=combined \
    aux_teacher_ar_noise_scale=1.0 \
    aux_teacher_ar_noise_alpha_max=10.0 \
    aux_teacher_ar_noise_seed_mask_chunks=1 \
    aux_teacher_ar_noise_burn_in_steps=20 \
    aux_teacher_ar_noise_variance_renorm=true \
    aux_teacher_ar_noise_variance_ema_decay=0.99 \
    max_gradient_chunks=2 \
    run_name=action_forcing_phase3_online_teacher_NR_G_v14_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
