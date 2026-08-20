#!/bin/bash
HOLDER=${HOLDER:?set HOLDER}
#!/bin/bash
# ---------------------------------------------------------------------
# 2026-08-18 FIXES for the AR arms (klar / msear / kldual / msedual):
#   1. dmd_ar_head_commit=student (was the CODE default "gt"). Set
#      EXPLICITLY here so it lands in the wandb config dump instead of
#      being an invisible code default. Every commit="gt" arm collapsed:
#      msedual m_fake +90.4% in its last bin and worse on 40/40 of the
#      final 40 steps vs its matched mse twin, video at 791 = flat noise;
#      klar/kldual stuttered by step 31, bad by 46 (faster because klar is
#      AR-ONLY, no TF head to dilute the AR gradient). GT commit conditions
#      the teacher on a past the student never has -> off-manifold,
#      unreachable target, AND it leaks the band's GT into a conditional
#      that is meant to be past-only.
#      IMPORTANT: this does NOT remove GT context. The AR prefill still
#      feeds clean GT context (ctx[:, s0:s0+npb] = the 42f gt_ctx), which
#      is the thing the probes showed helps (teacher on GT ctx 0.274 vs
#      student ctx 0.351 vs its OWN rolled ctx 0.352). Only the per-chunk
#      COMMIT source changes, to what the student actually has at inference.
#      Verify at runtime with gen/dmd_ar_commit_is_gt == 0.0.
#   2. aux_teacher_timestep_shift=1.0 -- with real_teacher_train_online=
#      false this key's ONLY live role is setting critic_timestep_shift
#      through a derived default (model/dmd_action_forcing.py:722-732).
#      Unset, the critic trains at the SCORING shift 5.0; the June trad run
#      that worked trained it at 1.0. Confirmed in both logs:
#        trad j5356442: "critic ... timestep_shift=1.0 (inherited from
#                        aux_teacher_timestep_shift)"
#        klar j6041847: "critic ... timestep_shift=None (inherited from
#                        timestep_shift (legacy default))"
#      Code rationale at :703-706: a critic trained at high t "modelled
#      p_fake poorly at the low/mid t where the 4-step student lives,
#      biasing (pred_real - pred_fake) toward blur."
# ---------------------------------------------------------------------
# =====================================================================
# DMD-10K STATIONARY (2026-08-16): reconnect phase-3 DMD to the 10k ODE
# students. Template = train_c8_stat_wave_dmd3_fwd_freal_ode1000.sbatch (the
# validated DMD<-ODE handoff precedent); deltas per
# .claude/dmd_gan_stage_reference.md §7:
#   * 14e lineage: v14e teacher LoRA + pca_raw actions ([0,1]) — the export
#     below is LOAD-BEARING (silent z2/z7 corruption without it).
#   * student rung ladder [1000,625,357.142857,208.333333] (rollkl 20-step
#     grid), NOT the phase-1 default.
#   * DMD ONLY: GAN off (backbone override kept: the constructor check fires
#     even with the GAN off), flash-DMD off, forward-noiser/CARN off,
#     stat anchor 0, teacher FROZEN (plain frozen v14e; no unlock).
#   * dmd_42f_clean_match OFF (in-flight dmd3 variant, deferred).
#   * strict_ode_load=true (default false swallows key drift).
#   * NOTE deferred: 9x int-rounded timesteps (<=0.04% shift, immaterial).
#
# 14e SERVE-CONTRACT ALIGNMENT (2026-08-17). Six DMD runs degraded the
# student identically. Root cause: 14e was trained through a REAL KV-cache
# AR rollout with a strict contract (action-forcing/af_model/ode_rollout.py)
# while DMD served it under a different one. 14d was teacher-forced with no
# cache at all, so DMD's regime was neutral for it — that is the 14d->14e
# delta.
#
# UNIFIED CONTRACT (block-relative RoPE) — one convention for BOTH
# stationary and rolling generation. The attention window and its rotation
# slots are FIXED; data flows through them (newest chunk -> newest slot,
# older chunks shift back, whatever leaves the window is evicted). Legal
# because RoPE attention only sees the RELATIVE offset (i-j) and the 14e
# student's attention was already capped at 21 frames: an evicted frame is
# one it could never have attended to. No bridge stage, no ODE retrain.
#   window (local_attn_size) = 21 frames = the student's trained window
#     (local_attn_chunks=7 x nfb=3, ode_rollout.py:202). The old
#     model_kwargs.local_attn_size=24 never actually reached the roll
#     logic: local_attn_size_schedule=[[0,21]] overwrites every module's
#     local_attn_size before the first sequence opens
#     (trainer:1866 -> _apply_attn_size_if_changed), so 24 survived ONLY as
#     max_attention_size (set once at pipeline construction from
#     model_kwargs and never updated by the schedule). max_attention_size
#     only slices [local_end - span : local_end], so a span WIDER than the
#     buffer reads nothing unwritten — it just attends the whole cache.
#     The six-run damage was therefore not rotation corruption but CONTENT
#     loss: buffer 21 frames vs a 9+3+21 = 33-frame sequence, so 12 frames
#     rolled out and the entire clean GT context was evicted mid-rollout.
#     model_kwargs.local_attn_size and the schedule are now both 21 so the
#     span, the roll trigger and the query anchor agree.
#   buffer (kv_cache_frames) >= window + npb, ALWAYS. The seed-aware sizing
#     (pipeline/action_forcing_training.py, kv_cache_frames) is
#       seed_prefill_frames + max(num_max_frames, rollout_frames) + npb
#       = 9 + 21 + 3 = 33 frames,  floored at window + npb = 21 + 3 = 24.
#     So buffer = 33 frames. The pipeline prints one [KV-CACHE] startup line
#     with window/buffer/npb and the frame at which rolling would begin.
#     window > buffer is a WARNING there, not a raise (2026-08-17): with the
#     buffer-relative RoPE anchor an over-large window shifts no offset, it
#     only caps the effective context at the buffer.
#   ARITHMETIC, CORRECTED (2026-08-17). The old header read
#     "cf(9) + anchor+rollout(3+21) = 33" and "never rolls (33-frame
#     sequence in a 33-frame buffer)" — that conflated the BUFFER formula
#     with the SEQUENCE length. With streaming_chunk_size=12 the sequence
#     actually written is
#       cf(9 seed) + anchor(3) + rollout(streaming_chunk_size = 12)
#       = 24 frames
#     in a 33-frame buffer, i.e. 9 frames of DELIBERATE SLACK (3 further
#     npb=3 chunks). The 21 in the buffer formula is num_max_frames (the
#     pipeline's window bound), NOT the number of frames this recipe rolls.
#     Conclusion is unchanged — it still never rolls — but with margin, and
#     the margin is now stated rather than accidental.
#   eviction is EXPECTED and safe — it only ever drops frames outside the
#     window. The old "buffer > window shifts every offset by
#     (window - buffer)" caveat is OBSOLETE: utils/infinity_rope.py now
#     anchors the query BUFFER-relative (num_cache_frames - num_new_frames,
#     the same origin K is rotated against), so rotation offsets are
#     continuous at any buffer depth, rolling or not. Deepening the buffer
#     is safe.
#   real seed at t=0 (seed_prefill_mode=real): write the real seed latents
#     into the cache (ode_rollout.py:315-325) instead of the student's own
#     noised-then-denoised ESTIMATE at context_noise. Implies the seed-aware
#     cache sizing (kv_cache_seed_headroom).
#   student commit: unchanged — the pipeline already commits the student's
#     own denoised_pred into the cache every chunk from step 0 (the Step 3.4
#     context-noise commit forward, pipeline/action_forcing_training.py:1574).
#   dmd_context_clean_frames=9 — 3 chunks of REAL seed, as the ODE stage
#     used (3 seed chunks x npb=3). The old 18 was NOT "what keeps the cache
#     from rolling": the cache size ignored the seed prefill entirely, so a
#     BIGGER cf made the roll happen SOONER, not later.
#   infinity_rope=true (config default) — the block-relative patch IS this
#     contract; it is also what utils/eval_causal_AR.py serves with, so
#     train/inference agree. In the never-roll regime it is mathematically
#     identical to the absolute causal_rope_apply path the 14e ODE stage
#     trained under (pre-fill rotation indices ARE the absolute ones), so
#     turning it on costs nothing and buys the rolling regime.
#     `cached_rope_action_aware` is only needed on the infinity_rope=false
#     branch (trainer/causal_action_forcing_train.py:1258-1313) and is inert
#     here — the patched forward never reads it.
#     Teacher/critic are unaffected either way: the TF scorers run with
#     kv_cache=None and the patched forward delegates straight to the
#     original there.
#   * commit source: unchanged — the pipeline already commits the student's
#     own denoised_pred into the cache every chunk from step 0
#     (pipeline/action_forcing_training.py:1921).
#   * 42f layout 3|3|1 (ns=4, gt_after=1): 3 GT context chunks | 3
#     supervised student chunks | 1 unsupervised student chunk. Derivation:
#     ns=4 -> num_sup=3 -> sup_frames=9; gt_after_chunks=1 -> 3 frames;
#     n_ctx = 21-9-3 = 9 frames = 3 chunks. OOM fallback to the old 3|2|2:
#     submit with D42_NS=3 D42_AFTER=2.
# Submit:
#   sbatch -J dmd10k-<tag> --export=ALL,DARM=<tag>,ODE_CKPT=<path>[,DEXTRA=...]
#     [-N 2 + MAXSTEPS=3 for smokes]  sbatch/train_dmd10k_stat.sbatch
# Judge by gen/dmd_mae_gate_m_fake (falling = student approaching teacher),
# real_score_mae_vs_gt, and teacher_match probes on saved ckpts.
# =====================================================================
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
: "${DARM:?submit with --export=ALL,DARM=<tag>,ODE_CKPT=<path>}"
: "${ODE_CKPT:?submit with --export=ALL,DARM=<tag>,ODE_CKPT=<path>}"
MAXSTEPS=${MAXSTEPS:-200}
DEXTRA=${DEXTRA:-}
# Space-free teacher-head switches. `sbatch --export=...,DEXTRA="a=1 b=2"`
# puts a SPACE inside one exported value, which is exactly the kind of thing
# that mangles silently and trains the wrong config. AR_HEAD / TF_HEAD carry
# a single token each and are assembled here instead.
#   AR_HEAD=1.0            -> dual head (TF + AR)
#   AR_HEAD=1.0 TF_HEAD=0  -> AR-only
[ -n "${AR_HEAD:-}" ] && DEXTRA="$DEXTRA dmd_ar_head_weight=${AR_HEAD}"
[ -n "${TF_HEAD:-}" ] && DEXTRA="$DEXTRA dmd_tf_head_weight=${TF_HEAD}"
echo "DMD10K head config: AR_HEAD=${AR_HEAD:-unset} TF_HEAD=${TF_HEAD:-unset} -> DEXTRA=[$DEXTRA]"

CONFIG=configs/action_forcing_phase3_dmd.yaml
LOGDIR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/dmd10k_${DARM}
WANDB_SAVE_DIR=wandb
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export ARRWM_ACTION_ENCODER=pca_raw
# TMPDIR/LOCALDIR HYGIENE (2026-08-17): submitted with --export=ALL, so the
# submitting shell's TMPDIR leaks in. An agent session sets TMPDIR to a
# session-local path (/local/user/<id>) that does NOT exist on the compute
# nodes -> PermissionError at rank init (job 6035826 died in 28s). The ODE-era
# drivers all pinned TMPDIR=/tmp; this one did not. Pin it here.
export TMPDIR=/tmp
unset LOCALDIR APPTAINER_CACHEDIR
# Scan-free start (2026-08-16): pre-built weunz manifest (v4, 2,549 rides
# >=69f, filtered from the .rec_scratch 21.9GB manifest) — skips the ~50-min
# per-launch ride scan entirely.
export ARRWM_MANIFEST_PICKLE=/scratch/u6ex/as1748.u6ex/ARRWM/analysis/.dmd_weunz_manifest_min69.pt
mkdir -p "$LOGDIR" logs
MASTER_ADDR=$(scontrol show hostnames "$(squeue -j $HOLDER -h -o %N)" | head -n1)
MASTER_PORT=$((29500 + HOLDER % 16000 + 911))
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn NCCL_DEBUG=WARN NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
echo "DMD10K arm=$DARM ckpt=$ODE_CKPT steps=$MAXSTEPS nodes=2 $(date)"

srun --jobid=$HOLDER --overlap --nodes=2 --ntasks-per-node=1 --gpus-per-node=4 torchrun \
  --nnodes=2 \
  --nproc_per_node=4 \
  --rdzv_id=klarh$HOLDER \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  trainer/causal_action_forcing_train.py \
  --config $CONFIG \
  --override \
    log_dir=$LOGDIR \
    wandb_dir=$WANDB_SAVE_DIR \
    wandb_project=longlive-phase-3 \
    max_steps=$MAXSTEPS \
    checkpoint_interval=${CKPT_EVERY:-100} \
    keep_last_n_checkpoints=99 \
    encoded_root=/projects/u6ex/fbots/frodobots_encoded_weunz \
    holdout_zarr_list=configs/holdout_rides.txt \
    holdout_eval_root=/projects/u6ex/fbots/frodobots_encoded_weu_holdout \
    holdout_eval_mode=cycle \
    sample_7chunk_enabled=false \
    max_ride_frames=900 \
    max_ride_frames_random=false \
    seed=1234 \
    action_teacher_mode="off" \
    action_critic_aux_enabled=false \
    state_probe_aux_enabled=false \
    generator_action_z_guidance_weight=0.0 \
    lora_action_critic_z_guidance_weight=0.0 \
    action_dims=[0,1] \
    action_critic_dims=[0,1] \
    denoising_step_list=[1000,625,357.142857,208.333333] \
    max_rolls_per_ride=1 \
    dfake_gen_update_ratio=5 \
    model_kwargs.local_attn_size=21 \
    num_workers=0 \
    sample_fps=16 \
    boundary_vae_roundtrip=false \
    local_attn_size_schedule=[[0,21]] \
    infinity_rope=true \
    seed_prefill_mode=real \
    kv_cache_seed_headroom=true \
    real_teacher_causal_mask=false \
    real_guidance_scale=0.0 \
    dmd_context_clean_frames=9 \
    streaming_chunk_size=12 \
    dmd_loss_weight=1.0 \
    dmd_loss_start_step=0 \
    dmd_loss_warmup_steps=20 \
    dmd_normalization_enabled=true \
    dmd_mae_gate_enabled=true \
    dmd_mae_gate_r_full=2.0 \
    dmd_mae_gate_ema=0.0 \
    dmd_mae_gate_min_weight=0.0 \
    dmd_mae_gate_exponent=0.75 \
    stat_anchor_loss_weight=${STAT_ANCHOR:-0.0} \
    stat_anchor_mode=gt_window \
    stat_anchor_match_k=2 \
    stat_anchor_M2_short_weight=0.1 \
    stat_anchor_M2_long_weight=0.0 \
    stat_anchor_TV_short_weight=0.1 \
    stat_anchor_TV_long_weight=0.0 \
    stat_anchor_M1_short_weight=3e-9 \
    stat_anchor_M1_long_weight=0 \
    stat_anchor_SOS_short_weight=0.0 \
    stat_anchor_SOS_long_weight=0.0 \
    stat_anchor_STD_short_weight=0.0 \
    stat_anchor_STD_long_weight=0.0 \
    stat_anchor_rel_tol_short=0.0 \
    stat_anchor_rel_tol_long=0.0 \
    dmd_ar_head_commit=student \
    aux_teacher_timestep_shift=1.0 \
    dmd_asymmetric_scoring_enabled=false \
    dmd_42f_enabled=true \
    dmd_42f_2chunk=false \
    dmd_42f_seed_last=false \
    dmd_42f_gt_after_chunks=${D42_AFTER:-1} \
    dmd_42f_rand_sup_slot=false \
    dmd_42f_allsup=false \
    dmd_42f_gt_anchor=true \
    dmd_42f_num_chunks=${D42_NS:-4} \
    dmd_42f_fix_clean_counterpart=false \
    dmd_42f_clean_drift_enabled=false \
    dmd_42f_clean_match_enabled=false \
    dmd_only_first_chunk_per_ride=true \
    flash_dmd_enabled=false \
    fake_score_ema_weight=0.95 \
    motion_start_threshold=5.0 \
    fake_score_lora_enabled=false \
    gan_enabled=false \
    gan_backbone=ladd_teacher_feat \
    gan_loss_weight=0.0 \
    real_teacher_train_online=false \
    dmd_frozen_teacher_pass_enabled=false \
    real_score_ema_weight=0.0 \
    aux_teacher_loss_weight=0.0 \
    aux_real_clean_x_source=gt \
    aux_noisy_from_raw_gt=true \
    aux_clean_x_random_window=false \
    use_8bit_adam=true \
    gen_gradient_checkpointing=true \
    forward_noiser_enabled=false \
    carn_recurse=false \
    forward_noiser_apply_gt_former=false \
    ladd_adjacent_chunks_enabled=false \
    real_score_gradient_checkpointing=true \
    fake_score_gradient_checkpointing=true \
    strict_ode_load=true \
    ode_generator_checkpoint=$ODE_CKPT \
    v14_teacher_checkpoint=/scratch/u6ex/as1748.u6ex/ARRWM/logs/v14e_pca8_raw/causal_lora_step0005000.pt \
    student_phase_lora_enabled=false \
    $DEXTRA \
    run_name=dmd10k_${DARM}_h${HOLDER}

echo "DMD10K $DARM completed $(date)"
