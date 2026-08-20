#!/bin/bash
#SBATCH --job-name=dmd3fgan
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
# DMD3 — the distillation recipe. THREE components, nothing else:
#
#   1. STAT ANCHOR  stat_anchor_loss_weight=1.0, mode=gt_window
#      Only M2_short=0.1 (variance/energy) and TV_short=0.1 (temporal
#      variation). M1 (mean) is hard 0.0 -- June ran it at 3e-9, i.e.
#      never actually tested WITH a mean constraint. STD/SOS/all _long = 0.
#      LOO verdict: the ONLY load-bearing component. Remove it and the run
#      collapses to black by ~step 61.
#
#   2. DISCRETE RUNG SAMPLING  dmd_sample_at_rungs=true
#      DMD scores at t drawn from the student's OWN ladder
#      [1000, 625, 357.142857, 208.333333] instead of a shifted continuum.
#      The 4-rung student only ever OCCUPIES those noise levels; sampling
#      between them asked the teacher about states it never visits.
#      This is the change that visibly improved samples.
#
#   3. TRAJECTORY DISTILLATION  dmd_real_traj_enabled=true (max_steps=6)
#      pred_real is no longer a single Euler step. Each rung integrates ITS
#      OWN SEGMENT of the 14e teacher's 20-step chain by deterministic
#      Euler on the same grid the teacher used to generate the LMDB data
#      (FlowMatchScheduler shift=5.0, sigma_min=0.0, extra_one_step=True;
#      utils/causal_chain_rollout.py:135-157, GL_STEPS=20). The ladder
#      subsamples that grid at indices [0,15,18,19], so:
#        rung 1000 -> idx 0->15 (6 forwards, capped from 15)
#        rung  625 -> idx 15->18 (3)
#        rung  357 -> idx 18->19 (1)
#        rung  208 -> idx 19 is last; the teacher's own to_final step is
#                     identically x - sigma*v = _convert_flow_pred_to_x0,
#                     so the one-step pred_real ALREADY is the endpoint (0)
#      The student therefore learns to make ONE jump that lands where the
#      teacher's many small steps land. Verified: logs dmd_traj_steps=3 at
#      t=625 / dmd_ar_traj_steps for the AR head.
#
# The MAE gate is DISABLED (weight always 1). That is an ABSENCE, not a
# component. m_real/m_fake/ratio are still logged -- _dmd_mae_gate_weight
# was changed to compute and log them even when disabled, forcing only the
# weight, so an ungated run is not blind.
#
# DROPPED after leave-one-out showed them INERT:
#   critic timestep shift (aux_teacher_timestep_shift) -- removing it made
#     no difference; also moot, the rung sampler returns before any shift.
#   band-local eq.8 normalizer -- removing it made no difference.
#   timestep_shift / min_score_timestep -- dead under rung sampling.
#
# GEOMETRY: 42f, clean half ALL GT, back-shifted one chunk (the "oracle" --
# every supervised student frame's exact GT counterpart sits at the matched
# rotary index). clean_drift/clean_match OFF. G1/G2/G5 moved it 1/2/3
# chunks forward and were indistinguishable from each other.
#
# OPEN QUESTION: tf91 (this recipe) looks no different to G9. Every
# ablation except the stat anchor has read null. That is the signature of
# ONE term dominating -- see the STAT_ANCHOR weight sweep.
# =====================================================================
# =====================================================================
# DMD-10K LOW-T PROBE (2026-08-18). Fork of train_dmd10k_stat.sbatch. The
# 800-step arms (mse 6040134 / kl 6040135) came out FLAT on a
# timestep-controlled regression, yet the student's per-latent std collapses
# to ~1/3 of GT within ~150 steps. Measured cause: the MAE gate is MONOTONE
# in t and so SELECTS FOR THE ORACLE REGIME.
#   t 20-209 : m_real 0.569  m_fake 0.592  ratio 1.046  gate w 0.095
#   t 901-981: m_real 0.188  m_fake 0.544  ratio 3.00   gate w 0.984
# pred_real is the teacher denoising the STUDENT's sample re-noised at t. At
# low t the noise does not hide the student's content, so the teacher hands
# back a lightly-cleaned copy -> nothing to teach. At high t the student's
# content is ERASED and the teacher REGENERATES from the clean GT half ->
# lands near GT -> big ratio -> gate opens. Effective gradient budget:
# t<=208 = 1.4%, t<=357 = 5.2%, t>800 = 67%. The student's ladder
# [1000,625,357,208] sets its final image at rungs 357/208 — which receive
# 5% of the DMD gradient.
#
# THREE DELTAS vs the stat arm (everything else byte-identical):
#   1. dmd_mae_gate_min_weight=1.0 (NOT dmd_mae_gate_enabled=false).
#      Both neutralise the gate (w == 1.0 always), but `enabled=false`
#      returns at model/dmd_action_forcing.py:4640 BEFORE computing
#      m_real / m_fake / ratio / student_mae_vs_gt — it would blind the
#      run's primary diagnostic. min_weight=1.0 keeps every metric and
#      makes w = max(1.0, min(1.0, base**exp)) = 1.0 unconditionally.
#   2. timestep_shift=0.5 (was 5.0). SINGLE knob. Top-level
#      `timestep_shift` is a DIFFERENT key from `model_kwargs.timestep_shift`
#      (model/base.py:202 reads the latter for the scorers' FlowMatch sigma
#      grid, which STAYS at 5.0) — verified `self.timestep_shift` is read
#      ONLY by `_sample_dmd_timestep`. t = s*u/(1+(s-1)u):
#        s=5.0  -> p25 529, med 804, p75 952  (68% above t=625)
#        s=0.5  -> p25 143, med 333, p75 600  (34% below 208, 23% above 625)
#        s=0.25 -> p25  77, med 200, p75 429  (51% below 208 — HALF the
#                  budget under the lowest rung, where ratio 1.046 says the
#                  teacher is measurably uninformative). Rejected.
#      0.5 centres the mass on rung 357 and spans all four rungs.
#   3. dmd_debug_step=208. Diagnostic ONLY (model/dmd_action_forcing.py:2968)
#      — runs an extra real/fake pair at fixed t=208 for the EVAL STASH; the
#      training gradient still uses the random t. This is the discriminating
#      measurement: if pred_real at t=208 still looks sharp, the problem is
#      purely BUDGET and the shift fixes it; if pred_real at t=208 looks like
#      the student's own degraded output (PREDICTED by ratio=1.046), the 42f
#      geometry yields no low-noise signal and a reweight ALONE WILL NOT FIX
#      IT.
#
# ts_schedule DELIBERATELY LEFT FALSE. It is INCOMPATIBLE with shift<1:
# `_get_timestep` samples uniform in [min,max], THEN applies the shift, THEN
# clamps to the GLOBAL [min_step,max_step]=[20,980] — not to [min,max]. With
# ts_schedule on and a rung range of [357,625], shift 0.5 maps it to
# [~217,~450], i.e. straight through the floor ts_schedule just asked for.
# The two knobs fight; pick one. (Also: phase1.yaml:563-571 says both
# ts_schedule flags were disabled deliberately for CF parity.)
#
# flash_dmd_enabled STAYS FALSE. It is NOT a logging flag: it adds one
# grad-on gen forward per block at t=60 whose output feeds the GAN/aux
# losses AND REASSIGNS `cache_pred = flash_dmd_pred.detach()`
# (pipeline/action_forcing_training.py:2229-2242), i.e. it changes what gets
# committed to the KV cache and therefore the rollout itself. Turning it on
# to de-confound the sample videos would silently change the training
# contract. Instead the exit rung is now LOGGED (gen/exit_rung_t_from|to,
# model/dmd_action_forcing.py) so the random-exit-rung renders are
# identifiable after the fact.
#
# JUNE PRECEDENT: timestep_shift=0.25 already existed as
# sbatch/train_stat_wave_dmd3_lowt.sbatch (Jun 18); the VALIDATED template
# train_c8_stat_wave_dmd3_fwd_freal_ode1000.sbatch used shift=5.0 and
# dmd_context_clean_frames=18 (we run 9).
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
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
MASTER_PORT=$((29500 + SLURM_JOB_ID % 16000))
export NCCL_CROSS_NIC=1 NCCL_SOCKET_IFNAME=hsn NCCL_DEBUG=WARN NCCL_IB_TIMEOUT=50
export OMP_NUM_THREADS=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,garbage_collection_threshold:0.8
echo "DMD10K arm=$DARM ckpt=$ODE_CKPT steps=$MAXSTEPS nodes=$SLURM_NNODES $(date)"

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
    dmd_mae_gate_enabled=false \
    dmd_mae_gate_r_full=2.0 \
    dmd_mae_gate_ema=0.0 \
    dmd_mae_gate_min_weight=0.0 \
    dmd_mae_gate_exponent=0.75 \
    dmd_sample_at_rungs=true \
    dmd_real_traj_enabled=false \
    dmd_real_traj_max_steps=6 \
    dmd_debug_step=208 \
    stat_anchor_loss_weight=${STAT_ANCHOR:?set STAT_ANCHOR=1.0 (or 0.0 to ablate deliberately)} \
    stat_anchor_mode=gt_window \
    stat_anchor_match_k=2 \
    stat_anchor_M2_short_weight=0.1 \
    stat_anchor_M2_long_weight=0.0 \
    stat_anchor_TV_short_weight=0.1 \
    stat_anchor_TV_long_weight=0.0 \
    stat_anchor_M1_short_weight=0.0 \
    stat_anchor_M1_long_weight=0 \
    stat_anchor_SOS_short_weight=0.0 \
    stat_anchor_SOS_long_weight=0.0 \
    stat_anchor_STD_short_weight=0.0 \
    stat_anchor_STD_long_weight=0.0 \
    stat_anchor_rel_tol_short=0.0 \
    stat_anchor_rel_tol_long=0.0 \
    dmd_ar_head_commit=student \
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
    flash_dmd_enabled=true \
    flash_dmd_gan_t=60 \
    gan_enabled=true \
    gan_backbone=ladd_teacher_feat \
    gan_loss_weight=1.0 \
    gan_lr=2e-5 \
    gan_critic_lr=0.0002 \
    gan_critic_hidden=512 \
    gan_critic_num_blocks=6 \
    gan_critic_grad_frames=3 \
    gan_critic_grad_full_every=50 \
    gan_critic_grad_loss_weight=1.0 \
    gan_critic_warmup_steps=20 \
    gan_disc_start_step=20 \
    gan_disc_base_channels=64 \
    gan_disc_in_channels=16 \
    gan_disc_num_blocks=4 \
    gan_updates_per_step=5 \
    gan_warmup_steps=25 \
    gan_warmup_shape=linear \
    gan_max_grad_norm=10.0 \
    gan_r1_gamma=2.0 \
    gan_r2_gamma=1.0 \
    gan_weight_decay=0 \
    gan_gate_couple_enabled=false \
    ladd_cmap_dim=64 \
    ladd_proj_dim=512 \
    ladd_disc_head_kernel=3 \
    ladd_disc_loss_weight=1.0 \
    ladd_disc_micro_batch_groups=2 \
    ladd_r1_gamma=10 \
    ladd_r1_num_samples=6 \
    ladd_r1_every_n_steps=8 \
    ladd_r1_once_per_step=true \
    ladd_r1_normalize_tokens=true \
    ladd_r1_sigma=0.01 \
    ladd_r2_gamma=0 \
    ladd_r2_every_n_steps=2 \
    ladd_r2_phase_offset=1 \
    ladd_use_csm=true \
    ladd_use_lateral_proj=false \
    ladd_use_prompt_cond=false \
    ladd_diff_aug_policy=flip \
    ladd_adjacent_chunks_enabled=false \
    ladd_gt_vs_fake_enabled=false \
    ladd_pairs_per_step=0 \
    ladd_pair_start_seed_boundary=true \
    ladd_wavelet_hf_enabled=${WAVELET:-false} \
    ladd_gt_transition_enabled=true \
    ladd_gt_vs_fake_enabled=false \
    ladd_adjacent_chunks_enabled=false \
    ladd_gt_transition_match=true \
    ladd_gt_transition_match_k=4 \
    ladd_gen_guidance_micro_batch_groups=4 \
    ladd_wavelet_hf_augment=${WAVELET:-false} \
    $DEXTRA \
    run_name=dmd10k_${DARM}_j${SLURM_JOB_ID}

echo "DMD10K $DARM completed $(date)"
