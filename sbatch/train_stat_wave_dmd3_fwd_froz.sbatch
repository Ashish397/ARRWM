#!/bin/bash
#SBATCH --job-name=af-stat-wave-dmd3-fwd-froz
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
# ============================ stat_wave ===============================
# stat_main + WAVELET in AUGMENT mode. Authoritative config summary:
#   GAN:  wavelet ON but AUGMENTING, not replacing — ladd_wavelet_hf_augment
#         adds the HF wave view to the RAW latent (x_noisy + wavelet_hf(x)) so
#         the disc keeps BOTH modalities (raw structure + HF detail). LL band
#         stays gated off (drop_ll=true, ll_weight=0.0) at the same rate as
#         before. This is the opposite of the old failure: replacing the raw
#         view with HF-only blinded the disc; augmenting preserves it.
#         eq OFF, matching ON (match_k=4); ALL generated transition pairs
#         supervised (pairs_per_step=0) via micro_batch_groups=2.
#   R1:   token-NORMALIZED (ladd_r1_normalize_tokens) so gamma is portable
#         -> gamma=10 on the normalized scale; fires 1-of-5 per step
#         (ladd_r1_once_per_step); sigma=0.01.
#   GAN volume: gan_loss_weight=1.0.  Disc CAPACITY: proj_dim 256 -> 512.
#   DMD:  dmd_42f_seed_last=FALSE (per user) — the noisy after-block is the
#         student's OWN rolled chunks (DETACHED/no-grad), not a GT scaffold,
#         so the supervised chunk is scored in the future-context it actually
#         sees at inference (student), not an OOD GT-anchored future. Noisy:
#         [ G G G | S S | s s ] (s = student, no-grad).
#         dmd_42f_fix_clean_counterpart=FALSE (per user, overriding my
#         concern) — the supervised chunk's clean counterpart is now GT, not
#         student, so clean = all GT. CAVEAT ON RECORD: the bidir scorer can
#         now read the band's GT in the clean half (a leak the flag was added
#         to kill) + clean/noisy halves disagree (OOD vs v14's identical-half
#         contract); the DMD target is ALREADY GT (gt_target). Testing both.
#   FN:   carn_recurse=true -> forward noiser trained rollout1->rollout2
#         (MODEL-based, no GT); applied to the GT former with
#         forward_noiser_apply_gt_drift_cap (per-row level capped BELOW the
#         served student's drift -> never a gradient toward MORE causal
#         noise; later/more-drifted latents allow higher former levels).
# Judge by gen/dmd_mae_gate_m_fake (student MAE), NOT the (d_real-d_fake) gap.
# >>> SMOKE FIRST: confirm normalized R1 grad_sq scale (gamma sane) + no OOM
#     from proj_dim=512 before the full run. <<<
# Lineage notes below are historical (titles/deltas may be superseded).
# =====================================================================
# ---- v8j3g3 = v8j3g2 + PRINCIPLED R1 FIX + lighter GAN weight ----
# 3-agent review of j3g2's "gap decay": NOT A BUG, and the decay is BENIGN
# (the student CONVERGES to GT — as the gap collapsed 1.35->0.07, d_real
# ROSE 0.73->4.35 and student MAE m_fake HALVED). The disc matches each
# fake to its NEAREST GT, so an improving student shrinks the gap by
# construction. Judge j3g3 by gen/dmd_mae_gate_m_fake (student MAE, lower
# = better) and m_fake -> m_real, NOT by whether (d_real-d_fake) "holds".
# The one real issue: R1 was mis-scaled. The FD-R1 summed over ~47,000
# per-token logits so grad_sq ~1e6-3e7; with the old gamma the penalty
# (5e4-1.5e6) dwarfed the O(1) RpGAN signal -> disc over-regularized.
# g3 applies the PRINCIPLED fix (3 changes; the first two are new
# code, flag-gated default-off so other configs are byte-identical):
#   * ladd_r1_normalize_tokens=true: grad_sq now estimates ‖∇ MEAN_i D_i‖²
#     (token-count-INDEPENDENT) instead of ‖∇ Σ_i D_i‖². So gamma is now
#     portable/interpretable -> gamma=10.0 on the normalised scale (penalty
#     ~O(1e-2), well below RpGAN's O(0.7)) instead of a magic 1e-4.
#     >>> NOTE: SMOKE FIRST to read the logged normalised r3gan_r1_grad_sq
#         and confirm gamma=10 gives a sane penalty (not inert, not
#         dominant); finalise gamma from that. <<<
#   * ladd_r1_once_per_step=true: R1 fires on only the FIRST of the 5
#     disc-updates per fire-step (1-of-5), cutting the R1 dose ~5x (it
#     was firing on all 5).
#   * gan_loss_weight 1.5 -> 1.0 (per request). NOTE the tension: this
#     LOWERS how loud the GAN is on the generator; the R1 fix raises signal
#     QUALITY but to make the GAN a bigger DRIVER the weight should later
#     go UP (+ a DMD->GAN curriculum), not down.
# match_k reverted 12 -> 4 (per request). ALL generated fake transition
# pairs are still supervised (ladd_pairs_per_step=0); micro_batch_groups=2
# splits the n_fake fakes into 2 contiguous grad-accum groups that cover
# EVERY fake (verified: bounds partition [0,n_fake) with no drop).
# Everything else identical to v8j3g2 (eq off, wavelet off, all-pairs +
# FT_v3 micro-batch, match_k=12, matching on).
# =====================================================================
# ---- v8j3g2 = v8j3g + EQUALIZATION OFF (let the GAN supervise
#      brightness/magnitude) ----
# v8j3g kept the equalization stack ON (disc moment-blind, can only
# touch texture). Hypothesis for g2: with eq OFF the disc SEES the
# per-channel magnitude/brightness gap between student and GT, so the
# GAN can push the student to match GT brightness (not just texture).
# Smoke A (eq off, wave off) already confirmed the gap opens with eq
# off, so the disc still trains. WATCH-POINT: the e2 lineage added eq
# to stop a white/brightness COLLAPSE (~step 61: "brighter=real" ->
# student brightens -> runaway white). Mitigations still in place here:
# moment-preserving CARN restore (_carn_former) + rollout2-hot prebuild
# now skipped, so the original collapse driver is weakened. Monitor the
# clean_x_fake brightness over the first ~80 steps.
# ALSO folds in the R1 tuning from the v8j3g graph analysis: ladd_r1_gamma
# 0.02 -> 0.1 and ladd_r1_sigma 0.01 -> 0.05. j3g's gap opened (peak +0.93)
# but DECAYED (tail +0.19) while r3gan_r1_grad_sq exploded to ~4.5M (disc
# over-sharpening); stronger/smoother R1 should hold the gap open.
# AND maximises GAN pairs: ladd_gt_transition_match_k 4 -> 12. The disc
# already forwards the full match_max_real=12 unique GT reals (the cap is
# hit at k=4); scoring each of the 6 fake transition pairs against all 12
# (block-diagonal) gives 6x12=72 relativistic comparisons vs 6x4=24, at
# ~zero extra memory — the expansion is at the LOGIT level (index_select +
# repeat_interleave), same disc forward of <=12 reals + 6 fakes.
# Everything else identical to v8j3g (wavelet off, all-pairs supervision
# + FT_v3 micro-batch, matching on).
# =====================================================================
# ---- v8j3g = v8j3f + THE GAN FIX (wavelet off) + R1/DiffAug tweaks ----
# The j*f gt_transition disc gave ZERO signal (d_real==d_fake to 3
# decimals, d_loss==log2, gap=0). A 2x2 smoke factorial isolated the
# cause DECISIVELY: the WAVELET HF stage, not the equalization.
#   smoke A (eq off, wave off): gap +0.36  HEALTHY
#   smoke C (eq off, wave ON ): gap ~0.01  DEAD (common-mode drift)
#   smoke D (eq ON,  wave off): gap +0.36  HEALTHY
#   j*f     (eq ON,  wave ON ): gap  0     DEAD
# Pattern: wave ON -> dead, wave OFF -> healthy, INDEPENDENT of eq.
# Mechanism: ladd_wavelet_hf_drop_ll discards the LL (structure) band
# and the 48->16 adapter (xavier gain=0.1) bottlenecks the HF-only
# input, so the disc cannot form a separating direction -> it only
# shifts its common-mode bias. Equalization is INNOCENT for the gap
# (smoke D keeps it fully on and the gap opens), and it still buys the
# e2-lineage brightness-collapse protection, so it STAYS ON.
# Changes vs v8j3f:
#   * ladd_wavelet_hf_enabled: true -> FALSE   <-- THE FIX (disc now
#     sees the raw clean latent; also drops inert wavelet memory/compute).
#   * ladd_r1_every_n_steps: 1 -> 8  (LAZY R1; stops the every-step R1
#     grad-penalty spikes seen in j*f, grad_sq ~3e4). gamma unchanged.
#   * ladd_diff_aug_policy: flip,cutout,translation -> flip (cutout/
#     translation scramble the latent transition signal).
#   * ladd_pairs_per_step: 4 -> 0  = GAN-supervise ALL adjacent
#     transition pairs of the student rollout (7 chunks -> 6 pairs)
#     instead of a sampled 4. So every chunk the student generates gets
#     adversarial gradient, not a random subset.
#   * ladd_disc_micro_batch_groups=2 + ladd_r1_num_samples=6 (FT_v3's
#     disc-backward memory fix): split the n_fake disc forward+backward
#     into 2 grad-accum groups and subsample R1 to 6 unique reals, so the
#     larger pair count fits. Real side already bounded by
#     match_max_real<=12. Default-off elsewhere => byte-identical there.
# UNCHANGED: equalization stack (kept ON, proven harmless + protective)
# and nearest-twin matching (vindicated: smoke A/D opened the gap WITH
# matching on).
# =====================================================================
# ---- j-series add-on: ONLINE TEACHER + RANDOM-WINDOW clean_x curriculum
# (same block as v8j2, which beat h2): rank-32/alpha-64 LoRA teacher,
# EMA-0.95 as DMD real score, clean_x conditioning sliced from a random
# window of the same ride (drift 40->160; actions ride along), pairs=5 +
# 8-bit Adam + gen ckpt + separate backward to fit the aux pass. The
# base FN behaviour (CARN application on the GAN former) is unchanged.
# v8h3 = NAIVE FORWARD CARN-FORMER (h1 base, FN direction flipped back).
# The FN trains FORWARD (default, no reverse): clean GT @ level L ->
# student-drifted @ L, ONE conditioned call (carn_recurse=false +
# step_unconditioned=false: FN is given the drift level as carn_step).
# teacher_feat/SW loss. Application (forward_noiser_apply_gt_former=true):
# ONE FN call at carn_step=1 on the FORMER GT latent only of each matched
# real transition pair (moment-preserving / texture-only) -> the real
# anchor is [1-step-drifted GT former -> clean GT latter], the self-
# correcting transition e8's carn_former intended but never executed in
# the matched path. NAIVE on purpose: the SAME CARN'd former feeds both
# the D-update and the gen-side real view (no Req-1 drift cap, no Req-2
# gen-side clean former) — with one CARN step that leak is acceptable.
# vs h1: h1 = REVERSE FN applied to BOTH latents (de-CARN'd real);
# h3 = FORWARD FN applied to FORMER only (drifted-former transition).
# Confirmation in .err: [FN-GT-FORMER] ACTIVE ... carn_step=1.
#
# ---- legacy e-lineage notes below (mostly describe the FORWARD CARN) ----
# Phase-3 DMD one-step — v8e6 = v8e5 + gan_loss_weight=2.5 + 3h wall to
# reach step 200 in one shot (from scratch). Checkpoints now ALSO save the
# GAN disc + ForwardNoiser, so this run is cleanly resumable.
#
# ---- v8e5 base ----
# Phase-3 DMD one-step — v8e5 = v8e4 but FLATTEN (across all four) instead
# of per-pair.
#
# Same as v8e4 (per-channel mean + std cross-eq) EXCEPT
# ladd_gt_transition_xeq_preserve_delta=FALSE: instead of scaling each
# pair by one shared factor (which preserves the within-pair former->latter
# brightness/contrast transition), every one of the FOUR members -- real
# former, real latter, fake former, fake latter -- is normalized
# INDEPENDENTLY to the common per-channel mean a* and std s*. So the disc
# is FULLY moment-blind: no absolute level, no within-pair transition --
# pure texture only. This is the "across all four" variant vs v8e4's
# "per pair". per-channel + std stay ON.
# Config = v8e4 with preserve_delta=false.
#
# ---- v8e4/v8e3 lineage ----
# v8e4 = v8e3 + STD equalization; v8e3 = v8e2 + per-channel + preserve-delta.
#
# v8e2 confirmed cross-equalization kills the white-collapse (brightness
# flat past the old step-61 point). v8e2 flattened ALL FOUR pair members
# to one level -> the disc was fully brightness-blind (within-pair delta
# gone too). v8e3 flips on ladd_gt_transition_xeq_preserve_delta=true:
# each PAIR is scaled by ONE shared factor to the common level a*, so the
# real/fake ABSOLUTE level is still equalized (no collapse) but the
# within-pair former->latter brightness TRANSITION ratio is PRESERVED ->
# the disc can see, and push the student to match, GT's transition
# brightness behaviour. (To become the hardcoded default.)
# ALSO: ladd_gt_transition_xeq_per_channel=true -> the equalization
# magnitude is computed PER CHANNEL (reduce [F,H,W], keep C) instead of
# one scalar over all channels, so per-channel brightness/colour is fully
# non-discriminative. Fix code is shared; config = v8e2 + these two flags.
#
# ---- v8e2 lineage: CARN BRIGHTNESS-COLLAPSE FIX ----
#
# v8e collapsed to white by ~step 61. Root cause (transition GAN carn
# path): the FN trains on un-normalized rollout1->rollout2 (rollout2 runs
# HOT, +1 drift step) so FN(GT_former) drifts BRIGHTER; the GAN's
# mean_equalize was within-pool only (real and fake equalized separately),
# so the hot real pair sat above the fake pair -> disc learns
# "brighter=real" -> student brightens -> rollout2 brighter -> FN brighter
# -> runaway white. TWO fixes (both in shared trainer code; config is
# otherwise identical to v8e):
#   1) CROSS-EQUALIZED mean_equalize (now the hardcoded default): all four
#      members of each pair row -- real former+latter, fake former+latter
#      -- are scaled to ONE common per-row mean-magnitude. The disc can no
#      longer see ANY magnitude difference (within-pair OR real-vs-fake),
#      so the GAN affects only texture, never brightness. Target is
#      detached -> no magnitude gradient to the student.
#   2) MOMENT-PRESERVING carn (_carn_former): restore the former's
#      per-channel DC + magnitude after the FN, so carn changes only
#      texture STRUCTURE (no stats) and its texture isn't rescaled away by
#      the cross-eq. Belt-and-braces with (1).
#
# Inherits v8b2 (normalize-fix max(|f|,1) + MAE gate + GT-anchor +
# local_attn=24, R1-only GAN). Changes vs v8b2:
#   * Learned ForwardNoiser (CARN) trained to map rollout1 -> rollout2 =
#     the "+1 cartoon shift" (the per-step texture/detail drift of the
#     causal AR rollout). LATENT SPACE — NO wavelet decomposition.
#     forward_noiser_loss_mode=teacher_feat, but the match is now a
#     SLICED-WASSERSTEIN per-token distribution match in the frozen
#     teacher's feature space (NOT mean+std). v8d's mean+std match was
#     inert (fn_tf_loss ~1e-6) because GT<->student moments barely differ
#     and are pinned by the stat anchor; the full SW marginal captures the
#     texture/cartoon shift those moments missed. Step-unconditioned
#     (one generic +1 shift). Separate FN backward, FN optimizer only ->
#     fully DECOUPLED from fake_score.
#   * fake_alt_rollout2_num_seed_chunks=4 = (cf/npb)-2 = 6-2, so rollout2
#     is exactly TWO seed chunks shorter than rollout1 -> a +2 drift
#     step (matches the prebuild's max_length_r2 = max_length + npb
#     geometry). rollout2 auto-builds because loss_mode=teacher_feat.
#   * ladd_gt_transition_carn_former=true (1 step): apply the FN ONCE to
#     the FIRST GT chunk of each transition pair -> real anchor =
#     [cartoon-shifted GT former -> clean GT latter] = a SELF-CORRECTING
#     (de-cartoon) transition the student is pushed to match.
#   * GAN at v6e strength: gan_updates_per_step=5, ladd_pairs_per_step=7
#     (RESTORE GAN STRENGTH per the user's instruction). NOTE: rollout2's
#     second no-grad rollout is a real memory tax (it forced the 8d cuts);
#     5/7 + rollout2 may re-OOM. If it does, the lever is GAN strength
#     (5->3) or ladd_pairs_per_step, NOT the CARN.
#   * DMD-strength compensation for the gate kept: dmd_mae_gate_exponent=
#     0.75 (mild concave ramp -> DMD held above the linear weight, fading
#     toward parity, but gentler than the 0.5 sqrt ramp).
#
# 21->24 frame window, 200-step max. GAN ON (ladd_teacher_feat), R1-only.
# =====================================================================

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

CONFIG=configs/action_forcing_phase3_dmd.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase3_dmd_one_step
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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
export ARRWM_ROPE_DEBUG=1

echo "CONFIG=$CONFIG"
echo "LOGDIR=$LOGDIR"
echo "Starting Phase-3 DMD ONE-STEP (v8e = v8b2 + learned latent-space SW CARN, v6e GAN) on $(date)"
echo "  * allsup: 3 supervised chunks; cf=18 seed + GT-anchor = 7 GT; local_attn=24."
echo "  * 1 roll per ride (fresh seed each step)."
echo "  * GAN: ON (ladd_teacher_feat, matched gt_transition), R1-only, v6e strength (5/7)."
echo "  * CARN-former: learned FN, +1 cartoon shift (latent-space sliced-Wasserstein teacher-feat), rollout2 ON."
echo "  * dmd_normalization: ON, denom floor=1.0 -> divide by max(|f|,1)."
echo "  * MAE gate: ON (r_full=2.0, ema=0.0 [per-sample, no smoothing], min_weight=0.0, exponent=0.75)."
echo "  * Teacher: single frozen merged-v14 (no online, no aux pass)."
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
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_weu \
    holdout_zarr_list=configs/holdout_rides.txt \
    holdout_eval_root=/projects/u6ex/fbots/frodobots_encoded_weu_holdout \
    holdout_eval_mode=cycle \
    max_ride_frames=900 \
    max_ride_frames_random=true \
    action_teacher_mode=off \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    max_rolls_per_ride=1 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=24 \
    num_workers=0 \
    sample_fps=16 \
    boundary_vae_roundtrip=false \
    local_attn_size_schedule=[[0,24]] \
    real_teacher_causal_mask=false \
    real_guidance_scale=0.0 \
    dmd_context_clean_frames=18 \
    dmd_loss_weight=1.0 \
    dmd_loss_start_step=0 \
    dmd_loss_warmup_steps=20 \
    dmd_normalization_enabled=true \
    dmd_mae_gate_enabled=true \
    dmd_mae_gate_r_full=2.0 \
    dmd_mae_gate_ema=0.0 \
    dmd_mae_gate_min_weight=0.0 \
    dmd_mae_gate_exponent=0.75 \
    stat_anchor_loss_weight=1.0 \
    stat_anchor_STD_short_weight=0.0 \
    stat_anchor_STD_long_weight=0.0 \
    stat_anchor_mode=gt_window \
    stat_anchor_match_k=2 \
    stat_anchor_M2_short_weight=0.1 \
    stat_anchor_M2_long_weight=0.0 \
    stat_anchor_SOS_short_weight=0.0 \
    stat_anchor_SOS_long_weight=0.0 \
    stat_anchor_M1_short_weight=3e-9 \
    stat_anchor_M1_long_weight=0 \
    stat_anchor_TV_short_weight=0.1 \
    stat_anchor_TV_long_weight=0.0 \
    stat_anchor_rel_tol_short=0.0 \
    stat_anchor_rel_tol_long=0.0 \
    dmd_asymmetric_scoring_enabled=false \
    dmd_42f_enabled=true \
    dmd_42f_2chunk=false \
    dmd_42f_seed_last=false \
    dmd_42f_gt_after_chunks=2 \
    dmd_42f_rand_sup_slot=false \
    dmd_42f_allsup=false \
    dmd_42f_gt_anchor=true \
    dmd_42f_num_chunks=3 \
    dmd_42f_fix_clean_counterpart=false \
    dmd_42f_clean_drift_enabled=false \
    dmd_42f_clean_drift_start_step=40 \
    dmd_42f_clean_drift_full_step=160 \
    dmd_42f_clean_drift_couple_rope=false \
    dmd_42f_clean_match_enabled=true \
    dmd_42f_clean_match_max_drift_frames=40 \
    dmd_42f_clean_match_forward=true \
    dmd_only_first_chunk_per_ride=true \
    flash_dmd_enabled=true \
    fake_score_ema_weight=0.95 \
    motion_start_threshold=5.0 \
    fake_score_lora_enabled=false \
    gan_enabled=true \
    gan_backbone=ladd_teacher_feat \
    gan_loss_weight=1.0 \
    gan_lr=2e-5 \
    gan_critic_warmup_steps=20 \
    gan_disc_start_step=20 \
    gan_warmup_steps=25 \
    gan_warmup_shape=linear \
    gan_updates_per_step=5 \
    gan_r1_gamma=2.0 \
    gan_max_grad_norm=10.0 \
    ladd_proj_dim=512 \
    ladd_use_csm=true \
    ladd_use_lateral_proj=false \
    ladd_use_prompt_cond=false \
    ladd_cmap_dim=64 \
    ladd_disc_loss_weight=1.0 \
    ladd_r1_gamma=10.0 \
    ladd_r1_sigma=0.01 \
    ladd_r1_every_n_steps=8 \
    ladd_r1_normalize_tokens=true \
    ladd_r1_once_per_step=true \
    ladd_diff_aug_policy=flip \
    ladd_pairs_per_step=0 \
    ladd_disc_micro_batch_groups=2 \
    ladd_gen_guidance_micro_batch_groups=4 \
    ladd_r1_num_samples=6 \
    ladd_wavelet_hf_enabled=true \
    ladd_wavelet_hf_drop_ll=true \
    ladd_wavelet_hf_ll_weight=0.0 \
    ladd_wavelet_hf_adapter_init_gain=0.1 \
    ladd_wavelet_hf_augment=true \
    ladd_gt_vs_fake_enabled=false \
    ladd_gt_vs_fake_weight=0.0 \
    ladd_gt_transition_enabled=true \
    ladd_gt_transition_weight=1.0 \
    ladd_gt_transition_all_pairs=false \
    ladd_gt_transition_mean_equalize=false \
    ladd_gt_transition_xeq_preserve_delta=false \
    ladd_gt_transition_xeq_per_channel=false \
    ladd_gt_transition_n_real=0 \
    ladd_gt_transition_match=true \
    ladd_gt_transition_match_k=4 \
    real_teacher_train_online=true \
    dmd_frozen_teacher_pass_enabled=false \
    real_score_ema_weight=0.0 \
    real_teacher_lr=5e-5 \
    real_teacher_warmup_steps=0 \
    real_teacher_lr_decay_end_step=50 \
    aux_teacher_lora_rank=32 \
    aux_teacher_lora_alpha=64 \
    aux_teacher_loss_weight=1.0 \
    aux_teacher_timestep_shift=1.0 \
    real_teacher_anchor_lambda=0.003 \
    gan_gate_couple_enabled=true \
    aux_teacher_start_step=0 \
    aux_teacher_loss_warmup_steps=20 \
    aux_teacher_separate_backward=true \
    aux_real_clean_x_source=gt \
    aux_noisy_from_raw_gt=true \
    aux_clean_x_random_window=false \
    aux_clean_x_drift_start_step=40 \
    aux_clean_x_drift_full_step=160 \
    use_8bit_adam=true \
    gen_gradient_checkpointing=true \
    forward_noiser_enabled=true \
    forward_noiser_loss_mode=teacher_feat \
    forward_noiser_step_unconditioned=false \
    carn_recurse=true \
    forward_noiser_feat_t=0 \
    forward_noiser_sw_n_proj=64 \
    forward_noiser_sw_max_tokens=4096 \
    fake_alt_rollout2_num_seed_chunks=4 \
    forward_noiser_apply_gt_former=true \
    forward_noiser_apply_gt_level=1 \
    forward_noiser_apply_gt_drift_cap=true \
    ladd_adjacent_chunks_enabled=false \
    ladd_adjacent_chunks_weight=0.0 \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    run_name=stat_wave_dmd3_fwd_froz_j${SLURM_JOB_ID}

echo "Training completed on $(date)"
