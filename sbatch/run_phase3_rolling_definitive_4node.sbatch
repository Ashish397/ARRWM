#!/bin/bash
# Definitive rolling continuation of one matched Phase-3 stationary axis.
# Four nodes x four GPUs/node x one ride/rank = global batch size 16.
#
# PHASE3_PARENT_JOB names the matching successful stationary job. The
# launcher resumes its exact full phase1_step0000200.pt checkpoint in a new
# rolling run directory. Submit with afterok:<PHASE3_PARENT_JOB>.
#
# It may also be an ORDERED, comma- or space-separated list of candidate job
# ids ("<a>,<b>") for the primary/backup stationary pair. That pairs with a
# Slurm OR-dependency (afterok:<a>?afterok:<b>), which releases this job when
# EITHER parent completes: the launcher then selects the FIRST candidate whose
# checkpoint actually exists on disk. A single id is just a list of one and
# behaves exactly as before.
#
# Rolling local steps 0..999 correspond to absolute checkpoint steps
# 200..1199. Each optimizer step samples D context chunks, rolls D under
# no_grad, then generates one fresh seven-chunk terminal window and trains
# S1..S7 for DMD and the non-DMD CARN/GAN auxiliaries.
#SBATCH --job-name=p3roll4n
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=16:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -euo pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

: "${PHASE3_AXIS:?set PHASE3_AXIS=base|base_v2|base_stabilized|no_gan|no_commit|no_aux|no_action_critic|no_stat_anchor}"
SMOKE_MODE="${PHASE3_SMOKE:-0}"
if [ "$SMOKE_MODE" = "1" ]; then
  PHASE3_PARENT_JOB="${PHASE3_PARENT_JOB:-holder-smoke}"
else
  : "${PHASE3_PARENT_JOB:?set PHASE3_PARENT_JOB to the matching stationary job id}"
fi

case "$PHASE3_AXIS" in
  base|base_v2|base_stabilized|no_gan|no_commit|no_aux|no_action_critic|no_stat_anchor) ;;
  *)
    echo "[P3-ROLL-4N] invalid PHASE3_AXIS=$PHASE3_AXIS" >&2
    exit 2
    ;;
esac

EXPECTED_NODES="${PHASE3_EXPECTED_NODES:-4}"
if [ "${PHASE3_PRINT_ONLY:-0}" != "1" ]; then
  if [ "${SLURM_NNODES:-0}" -ne "$EXPECTED_NODES" ]; then
    echo "[P3-ROLL-4N] expected exactly $EXPECTED_NODES nodes; got ${SLURM_NNODES:-unset}" >&2
    exit 3
  fi
fi

SRC=sbatch/carncommit_long.sbatch
BUNDLE=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256/exact_residual_ladder_2808/decoder_shaped_all_residual_seed0.pt
EXPECTED_BUNDLE_SHA=de82ec3211ec408f34cca883be53c15d835e19d5a3860a0b7fe8840dad450357
EXPECTED_EXACT_STAGES=1,2,3,5,6,7,9,10,11,13,14,15
ARRWM_PYTHON=/scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python3.10

[ -f "$SRC" ] || {
  echo "[P3-ROLL-4N] missing source launcher: $SRC" >&2
  exit 4
}
[ -f "$BUNDLE" ] || {
  echo "[P3-ROLL-4N] missing Q1~0.99 bundle: $BUNDLE" >&2
  exit 4
}

ACTUAL_BUNDLE_SHA=$(sha256sum "$BUNDLE" | awk '{print $1}')
if [ "$ACTUAL_BUNDLE_SHA" != "$EXPECTED_BUNDLE_SHA" ]; then
  echo "[P3-ROLL-4N] Q1~0.99 bundle SHA mismatch: $ACTUAL_BUNDLE_SHA" >&2
  exit 4
fi

BUNDLE_STAGES=$("$ARRWM_PYTHON" - "$BUNDLE" <<'PY'
import sys
import torch

bundle = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if bundle.get("kind") != "wan_decoder_shaped_pullback_bundle":
    raise SystemExit("wrong decoder-pullback bundle kind")
if len(bundle.get("models", ())) != 17:
    raise SystemExit("decoder-pullback bundle must contain 17 operators")
if not bool(bundle.get("exact_fixed_stages", False)):
    raise SystemExit("decoder-pullback bundle must keep fixed stages exact")
print(",".join(str(int(value)) for value in bundle["exact_residual_stages"]))
PY
)
if [ "$BUNDLE_STAGES" != "$EXPECTED_EXACT_STAGES" ]; then
  echo "[P3-ROLL-4N] wrong exact residual stages: $BUNDLE_STAGES" >&2
  exit 4
fi

# Same integrated recipe as the matched stationary parent. Auto-resume is
# deliberately true: the exact parent checkpoint is linked into this new
# run directory below before torchrun starts.
RUN_MAX_STEPS="${PHASE3_MAX_STEPS_OVERRIDE:-5000}"
BASE_COMMON="max_steps=$RUN_MAX_STEPS sample_interval=15 checkpoint_interval=200 keep_last_n_checkpoints=6 auto_resume=true strict_resume_load=true resume_load_fake_score=true save_full_checkpoint=true log_interval=5 memory_audit_enabled=true empty_cache_interval=20 gan_grad_telemetry_every=5 texture_tripwire_every=5 gan_pixel_texture_enabled=false gan_of_enabled=false fake_alt_head_enabled=false real_teacher_train_online=false dmd_frozen_teacher_pass_enabled=false real_teacher_causal_mask=false fake_score_causal_mask=false boundary_vae_roundtrip=false state_probe_aux_enabled=false surrogate_critic_enabled=false surrogate_decoder_shaped_enabled=true surrogate_decoder_shaped_bundle=$BUNDLE surrogate_decoder_shaped_audit_every=0 surrogate_decoder_fresh_disc_order=true surrogate_teacher_backbone=ladd_pixel pix_gan_weight=0.0175 pix_r1_gamma=0.0 pix_crop_lat=[24,32] pix_lat_frames_per_crop=3 pix_finish_grad_enabled=true exit_exclude_last_rung=true pix_flash_grad_select_enabled=true ladd_feature_source=vgg ladd_fake_backbone_trainable=false ladd_defer_disc_update=false ladd_disc_loss_weight=0.0 ladd_r1_gamma=1 ladd_use_prompt_cond=false ladd_cmap_dim=0 ladd_use_action_cond=false ladd_action_cmap_dim=64 ladd_action_mismatch_weight=0.0 ladd_disc_micro_batch_groups=8 ladd_pixel_crop_rows=24 ladd_pixel_crop_cols=32 ladd_pixel_crops_per_row=2 ladd_pixel_lat_frames=3 ladd_pixel_frames_per_crop=5 ladd_pixel_border_trim=8 ladd_pixel_decode_batch=4 ladd_pixel_decode_split=1 ladd_pixel_decode_cache=false ladd_pixel_crop_stratify_x=1 ladd_pixel_crop_y_lo_frac=0.0 ladd_pixel_crop_y_hi_frac=1.0 ladd_pixel_encoder_trainable=false ladd_pixel_encoder_lr_scale=0.0 ladd_pixel_input_filter=raw_grad_hp ladd_pixel_swt_strength=0.0 stat_anchor_loss_weight=1.0 gan_lr=1e-3 gan_updates_per_step=1 ladd_fake_sample_source=commit ladd_disc_force_clean=true flash_dmd_enabled=true flash_dmd_gan_t=60"

# Paired F/G CARN continued from the stationary checkpoint. CARN is consumed
# only by aux-minus and committed-memory correction; every transition-GAN
# route is explicitly disabled because the +former / bridge / -latter family
# is not production-ready. With transition off, GT-vs-fake receives the full
# unit pair-mode budget. GAN and recurrence consume the SAME post-CARN
# endpoint.  The commit enters rolling at alpha=.0625 (absolute step 200),
# reaches a meaningful .125 at step 250, and is bounded to a 1.5% per-chunk
# displacement. This retains the commit benefit while preventing the measured
# 2.4--3.5% recurrent jumps from compounding under the GAN distribution.
CARN_COMMON="forward_noiser_enabled=true forward_noiser_train_source=rollout forward_noiser_rollout2_source=legacy forward_noiser_reverse=false forward_noiser_cycle_enabled=true cycle_rev_input_mode=paired_target cycle_freeze_g_in_cycle=true forward_noiser_pair_loss_weight=0.25 carn_recurse=false forward_noiser_apply_decoupled=false forward_noiser_apply_gt_former=false forward_noiser_apply_gt_both=false forward_noiser_apply_in_aux=false reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_level=1 reverse_noiser_dedrift_min_level=1 reverse_noiser_dedrift_steps=1 reverse_noiser_dedrift_alpha0=1.0 reverse_noiser_preserve_moments=true reverse_noiser_internalize_weight=0.25 reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_dedrift_apply_to_flash=false reverse_noiser_dedrift_apply_to_commit=true reverse_noiser_commit_use_absolute_level=true reverse_noiser_commit_alpha=0.125 reverse_noiser_commit_start_step=150 reverse_noiser_commit_ramp_steps=100 reverse_noiser_commit_max_relative_shift=0.015 gan_carn_commit_align_enabled=true gan_carn_commit_straight_through=true gen_aux_losses_x0_source=commit ladd_gt_vs_fake_enabled=true ladd_gt_vs_fake_weight=1.0 ladd_gt_transition_enabled=false ladd_gt_transition_weight=0.0 ladd_gt_transition_carn_bridge=false ladd_gt_transition_carn_former=false ladd_gt_transition_carn_latter_reverse=false ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=false ladd_gt_transition_match=false ladd_adjacent_chunks_enabled=false stat_anchor_use_clean_match_offset=false stat_anchor_match_k=2 action_critic_aux_enabled=true action_critic_freeze=true action_teacher_mode=off teacher_action_encoder=pca_raw generator_action_z_guidance_weight=0.3 lora_action_critic_z_guidance_weight=0.0 critic_updates_per_step=1 action_critic_z_loss_weight=0.25 critic_lr=3e-4"

# Random context curriculum, inclusive bounds:
#   local [0,100): 4..10; [100,200): 4..13; ...; +increment every 100 steps.
# With increment=3 the upper bound is 10+3*48 = 154 at local step 4800
# (absolute 5000), so cap=160 is NON-BINDING for the whole 5000-step run --
# the curriculum keeps extending for the life of the job instead of
# saturating at 37 by local step 900. Every axis, including base, uses this
# same +3 curriculum; changing depth and loss composition together would
# invalidate the ablation.
#
# Depth is the MIN of four independent limits (audited 2026-08-29):
#   max_rolls_per_ride, rolling_window_curriculum_cap, and
#   rolling_random_depth_max all clamp it in trainer/causal_action_forcing_train.py
#   (lines 20079 and 20100), so all three must move together; the fourth is
#   the physical per-ride capacity clamp (line 20206), which derives from
#   streaming_max_length via can_generate_more (model/dmd_action_forcing.py
#   line 13881) and MIN-reduces across ranks.
# rollout_frames/streaming_max_length reserve max context 160 + terminal 7
# = 167 chunks = 501 post-seed frames in the ride window. The KV cache does
# NOT scale with this: pipeline/streaming_training.py line 271 sizes it as
# (local_attn_size + slice_last_frames) * frame_seq_length, independent of
# streaming_max_length, so it FIFO-rolls at fixed memory and extra depth
# costs only no_grad rollout TIME plus a linear ride-window tensor
# (~0.19 MB/frame, 132 -> 501 frames = +70 MB/rank).
# Ceiling rationale: max_ride_frames=900 (carncommit_long.sbatch line 227)
# minus dmd_context_clean_frames=9 caps any ride at 891 frames = 297 chunks,
# i.e. 290 context chunks -- but driving streaming_max_length that high
# forces s_local_max=0 (trainer/toothpaste.py line 92: ride_len-cf-cap),
# pinning every ride's motion-aware seed offset to 0 and killing offset
# diversity. 501 keeps s_local_max=390 on a 900-frame ride, so 160 is the
# largest depth that preserves seed-offset diversity.
ROLLING_COMMON="num_training_frames=21 rollout_frames=501 streaming_chunk_size=3 streaming_min_new_frame=3 streaming_max_length=501 num_chunks_roll_forward=1 max_rolls_per_ride=160 rolling_window_per_step_enabled=true rolling_random_depth_enabled=true rolling_random_depth_min=4 rolling_random_depth_max=160 rolling_window_curriculum_start_step=200 rolling_window_curriculum_stage_steps=100 rolling_window_curriculum_first_max=10 rolling_window_curriculum_increment=3 rolling_window_curriculum_cap=160 rolling_toothpaste_enabled=false rolling_rolls_schedule_every=0 ftv3_postbuild_enabled=false streaming_mae_collapse_threshold=0 dmd_supervise_roll_mode=all dmd_only_first_chunk_per_ride=true dmd_42f_enabled=true dmd_42f_num_chunks=7 dmd_42f_seed_last=false dmd_42f_gt_after_chunks=1 dmd_42f_gt_anchor=false dmd_42f_rand_sup_slot=false dmd_42f_allsup=false dmd_42f_rolling_sup_new=false dmd_42f_allroll_student_ctx=false dmd_42f_clean_match_enabled=false dmd_42f_clean_match_forward=false dmd_42f_clean_match_forward_only=false aux_train_trailing_chunk=true dmd_42f_supervise_last=true dmd_rolling_ctx_last_rung=false sample_7chunk_enabled=true max_ride_frames_random=false"

ABLATION_EXTRA=""
COMMITDEDRIFT_ENV=true
case "$PHASE3_AXIS" in
  base)
    # Headline arm uses the same +3-context-chunks/100-step curriculum as
    # every ablation. The former +6 override confounded the experiment.
    ABLATION_EXTRA=""
    ;;
  base_v2)
    # Redo of base with the STANDARD +3-context-chunks/100-steps curriculum:
    # no increment override, so rolling_window_curriculum_increment=3 from
    # ROLLING_COMMON stands, exactly like every non-base axis.
    ABLATION_EXTRA=""
    ;;
  base_stabilized)
    # Main-only, post-hoc robustness continuation. Threshold 4.0 was fixed
    # from the four concurrently running ablations BEFORE this restart:
    # pooled stat_anchor_total n=82, median=.595, p90=2.011, p95=3.827.
    # The objective is exactly unchanged at/below 4.0. Above it the model's
    # rational tail keeps training on the SAME window while attenuating the
    # stat gradient by (4/L)^2. This is a new, explicitly named treatment;
    # the four ablations remain loaded/running and therefore untouched.
    ABLATION_EXTRA="stat_anchor_tail_threshold=4.0"
    ;;
  no_gan)
    # Keep action guidance on the same commit-aligned surface as base so GAN
    # remains the only ablated force.
    ABLATION_EXTRA="gan_enabled=false gan_loss_weight=0.0 surrogate_decoder_shaped_enabled=false surrogate_decoder_fresh_disc_order=false pix_gan_weight=0.0 ladd_gt_vs_fake_enabled=false ladd_gt_transition_enabled=false ladd_adjacent_chunks_enabled=false gan_pixel_texture_enabled=false gan_of_enabled=false"
    ;;
  no_commit)
    ABLATION_EXTRA="reverse_noiser_dedrift_apply_to_commit=false reverse_noiser_commit_alpha=0.0"
    COMMITDEDRIFT_ENV=false
    ;;
  no_aux)
    ABLATION_EXTRA="reverse_noiser_internalize_weight=0.0"
    ;;
  no_action_critic)
    # Blank the critic checkpoint too (last-wins over carncommit_long.sbatch) so
    # the loader skips its strict load instead of raising with no critic head.
    ABLATION_EXTRA="action_critic_aux_enabled=false action_teacher_mode=off generator_action_z_guidance_weight=0.0 lora_action_critic_z_guidance_weight=0.0 action_critic_checkpoint="
    ;;
  no_stat_anchor)
    ABLATION_EXTRA="stat_anchor_loss_weight=0.0"
    ;;
esac

SMOKE_EXTRA=""
if [ "$SMOKE_MODE" = "1" ]; then
  # Holder-only GPU path check. Depth is pinned to the curriculum lower bound
  # so the test exercises context warmup plus one terminal seven-chunk update.
  SMOKE_MAX_STEPS="${PHASE3_SMOKE_MAX_STEPS:-1}"
  SMOKE_ROLL_DEPTH="${PHASE3_SMOKE_ROLL_DEPTH:-4}"
  SMOKE_ROLL_DEPTH_MIN="${PHASE3_SMOKE_ROLL_DEPTH_MIN:-$SMOKE_ROLL_DEPTH}"
  SMOKE_ROLL_DEPTH_MAX="${PHASE3_SMOKE_ROLL_DEPTH_MAX:-$SMOKE_ROLL_DEPTH}"
  SMOKE_SAMPLE_INTERVAL="${PHASE3_SMOKE_SAMPLE_INTERVAL:-0}"
  SMOKE_DISABLE_WANDB="${PHASE3_SMOKE_DISABLE_WANDB:-true}"
  SMOKE_SAMPLE_7CHUNK="${PHASE3_SMOKE_SAMPLE_7CHUNK:-false}"
  SMOKE_EXTRA="max_steps=$SMOKE_MAX_STEPS sample_interval=$SMOKE_SAMPLE_INTERVAL checkpoint_interval=999999 keep_last_n_checkpoints=1 auto_resume=false strict_resume_load=false resume_load_fake_score=false save_full_checkpoint=false disable_wandb=$SMOKE_DISABLE_WANDB log_interval=1 dmd_loss_warmup_steps=0 sample_7chunk_enabled=$SMOKE_SAMPLE_7CHUNK rolling_random_depth_min=$SMOKE_ROLL_DEPTH_MIN rolling_random_depth_max=$SMOKE_ROLL_DEPTH_MAX rolling_window_curriculum_start_step=0"
fi

# Explicitly last, and empty by default.  This is reserved for labeled
# same-parent holder counterfactuals/telemetry; production axes do not set it.
PROBE_EXTRA="${PHASE3_EXTRA_OVERRIDES:-}"
COMMON="$BASE_COMMON $CARN_COMMON $ROLLING_COMMON $ABLATION_EXTRA $SMOKE_EXTRA $PROBE_EXTRA"
export COMMON

DARM="${PHASE3_DARM_OVERRIDE:-phase3_rolling_definitive_4n_${PHASE3_AXIS}_gbs16_2908}"
export DARM
if [ "$SMOKE_MODE" = "1" ]; then
  MAXSTEPS="$SMOKE_MAX_STEPS"
else
  MAXSTEPS="$RUN_MAX_STEPS"
fi
export MAXSTEPS
if [ "$SMOKE_MODE" = "1" ]; then
  CKPT_EVERY=999999
else
  CKPT_EVERY=200
fi
export CKPT_EVERY

if [ "$SMOKE_MODE" = "1" ]; then
  PARENT_CKPT="none (fresh holder smoke)"
else
  # Holder diagnostics may resume an earlier full stationary checkpoint to
  # test whether a stabilization prevents (rather than merely masks) a later
  # failure. Production leaves this unset and remains exactly step 200.
  PARENT_STEP="${PHASE3_PARENT_STEP_OVERRIDE:-200}"
  if ! [[ "$PARENT_STEP" =~ ^[0-9]+$ ]]; then
    echo "[P3-ROLL-4N] invalid PHASE3_PARENT_STEP_OVERRIDE=$PARENT_STEP" >&2
    exit 2
  fi
  PARENT_STEP_PADDED=$(printf '%07d' "$PARENT_STEP")
  # base_v2 is a redo of the base rolling recipe, so it continues from the SAME
  # stationary base parent; there is no stationary base_v2 run. Only the parent
  # axis is remapped -- the rolling DARM/run name keeps base_v2 and stays a
  # distinct run directory.
  PARENT_AXIS="$PHASE3_AXIS"
  if [ "$PHASE3_AXIS" = base_v2 ] || [ "$PHASE3_AXIS" = base_stabilized ]; then
    PARENT_AXIS=base
  fi
  PARENT_DARM="${PHASE3_PARENT_DARM_OVERRIDE:-phase3_stationary_definitive_8n_${PARENT_AXIS}_gbs32_2808}"
  # PHASE3_PARENT_JOB may be an ORDERED list of candidate stationary job ids
  # (comma- or space-separated) so one rolling job can hang off an
  # afterok:<a>?afterok:<b> OR-dependency over a primary/backup pair: Slurm
  # releases this job when EITHER parent completes, and only the one that
  # actually ran has a checkpoint on disk. Take the FIRST candidate whose
  # requested full checkpoint exists. A single id is a list of one, so this
  # resolves to exactly the same path as before. When NO candidate has a
  # checkpoint, PARENT_CKPT keeps the FIRST candidate's path so the existing
  # missing-checkpoint guard below still reports a concrete path and exit 6.
  PARENT_CKPT=""
  for _cand in $(printf '%s' "$PHASE3_PARENT_JOB" | tr ',' ' '); do
    _cand_ckpt="/scratch/u6ex/as1748.u6ex/ARRWM/logs/dmd10k_${PARENT_DARM}/dmd10k_${PARENT_DARM}_j${_cand}/phase1_step${PARENT_STEP_PADDED}.pt"
    if [ -z "$PARENT_CKPT" ]; then
      PARENT_CKPT="$_cand_ckpt"
      PARENT_JOB_SELECTED="$_cand"
    fi
    if [ -f "$_cand_ckpt" ]; then
      PARENT_CKPT="$_cand_ckpt"
      PARENT_JOB_SELECTED="$_cand"
      break
    fi
  done
  PARENT_RUN="dmd10k_${PARENT_DARM}_j${PARENT_JOB_SELECTED}"
fi

N_RUN_NAME=$(grep -c '^[[:space:]]*run_name=dmd10k_' "$SRC")
if [ "$N_RUN_NAME" -ne 1 ]; then
  echo "[P3-ROLL-4N] expected one run_name injection site; got $N_RUN_NAME" >&2
  exit 5
fi

TMP_LAUNCH=$(mktemp "/tmp/p3roll4n_${PHASE3_AXIS}_${SLURM_JOB_ID:-print}.XXXXXX")
trap 'rm -f "$TMP_LAUNCH"' EXIT
awk '
  /^[[:space:]]*run_name=dmd10k_/ { print "    " ENVIRON["COMMON"] " \\" }
  { print }
' "$SRC" > "$TMP_LAUNCH"
bash -n "$TMP_LAUNCH"

echo "[P3-ROLL-4N] axis=$PHASE3_AXIS parent_job=$PHASE3_PARENT_JOB nodes=${SLURM_NNODES:-$EXPECTED_NODES} ranks_per_node=4 global_batch=$((EXPECTED_NODES * 4)) max_steps=$MAXSTEPS time_limit=16:00:00 smoke=$SMOKE_MODE"
echo "[P3-ROLL-4N] random_context=min4 first_max10 increment=3 cap160 terminal_window=7 supervise=S1..S7"
echo "[P3-ROLL-4N] samples=15 checkpoint_interval=200 save_full_checkpoint=true"
echo "[P3-ROLL-4N] gan_pairs=gt_vs_fake:$([ "$PHASE3_AXIS" = no_gan ] && printf off || printf 1.0) transition:off adjacent:off carn_consumers=aux_minus+commit"
echo "[P3-ROLL-4N] parent_candidates=[$PHASE3_PARENT_JOB] parent_selected=${PARENT_JOB_SELECTED:-n/a}"
echo "[P3-ROLL-4N] parent_checkpoint=$PARENT_CKPT"
echo "[P3-ROLL-4N] bundle_sha=$ACTUAL_BUNDLE_SHA exact_residual_stages=[$BUNDLE_STAGES]"
echo "[P3-ROLL-4N] DARM=$DARM"
echo "[P3-ROLL-4N] last_wins=[$COMMON]"

if [ "${PHASE3_PRINT_ONLY:-0}" = "1" ]; then
  exit 0
fi

if [ "$SMOKE_MODE" != "1" ] && [ ! -f "$PARENT_CKPT" ]; then
  echo "[P3-ROLL-4N] required parent checkpoint missing: $PARENT_CKPT" >&2
  exit 6
fi

if [ "$SMOKE_MODE" != "1" ]; then
  ROLL_LOGDIR="/scratch/u6ex/as1748.u6ex/ARRWM/logs/dmd10k_${DARM}"
  ROLL_RUN="dmd10k_${DARM}_j${SLURM_JOB_ID}"
  ROLL_RUN_DIR="$ROLL_LOGDIR/$ROLL_RUN"
  RESUME_LINK="$ROLL_RUN_DIR/phase1_step${PARENT_STEP_PADDED}.pt"
  mkdir -p "$ROLL_RUN_DIR"
  if [ -L "$RESUME_LINK" ]; then
    if [ "$(readlink -f "$RESUME_LINK")" != "$(readlink -f "$PARENT_CKPT")" ]; then
      echo "[P3-ROLL-4N] existing resume link points at the wrong parent: $RESUME_LINK" >&2
      exit 6
    fi
  elif [ -e "$RESUME_LINK" ]; then
    echo "[P3-ROLL-4N] refusing to overwrite existing non-link: $RESUME_LINK" >&2
    exit 6
  else
    ln -s "$PARENT_CKPT" "$RESUME_LINK"
  fi
fi

# In-allocation launch-race retry. The GH200 launch race strands 38-75GB of
# unattributable HBM on 1-3 GPUs and OOMs 2-7 min in; the stranded memory is
# transient (freed when the failed attempt's processes die), so relaunching
# torchrun on the SAME allocation with a fresh rendezvous port
# (ARRWM_MASTER_PORT is read by carncommit_long.sbatch line ~186) usually
# wins the next coin flip without losing the nodes. Only failures faster
# than PHASE3_INPLACE_WINDOW qualify (race failures die in minutes; a late
# crash is a real bug and a from-scratch rerun would waste GPU-hours).
# After PHASE3_MAX_INPLACE_ATTEMPTS, fall back to scontrol requeue (fresh
# nodes, same job id / run dir / afterok children), capped via
# SLURM_RESTART_COUNT as before.
attempt=0
max_inplace="${PHASE3_MAX_INPLACE_ATTEMPTS:-3}"
inplace_window="${PHASE3_INPLACE_WINDOW:-1500}"
while :; do
  attempt=$((attempt+1))
  export ARRWM_MASTER_PORT=$((29500 + (SLURM_JOB_ID + attempt * 131) % 16000))
  launch_t0=$SECONDS
  set +e
  COMMITDEDRIFT=$COMMITDEDRIFT_ENV bash "$TMP_LAUNCH"
  rc=$?
  set -e
  [ "$rc" -eq 0 ] && break
  launch_dt=$((SECONDS - launch_t0))
  if [ "$launch_dt" -lt "$inplace_window" ] && [ "$attempt" -lt "$max_inplace" ]; then
    echo "[P3-ROLL-4N] attempt $attempt failed rc=$rc after ${launch_dt}s -> in-allocation relaunch with fresh rdzv port"
    srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c 'pkill -9 -f "[c]ausal_action_forcing_train.py" 2>/dev/null; sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed "s/^/[hbm-drain $(hostname)] /"' || true
    sleep 45
    continue
  fi
  if [ "$launch_dt" -ge "$inplace_window" ]; then
    echo "[P3-ROLL-4N] attempt $attempt failed rc=$rc after ${launch_dt}s (>= ${inplace_window}s window): not a launch race, giving up" >&2
    exit "$rc"
  fi
  restarts="${SLURM_RESTART_COUNT:-0}"
  max_requeue="${PHASE3_MAX_REQUEUE:-4}"
  if [ -n "${SLURM_JOB_ID:-}" ] && [ "$restarts" -lt "$max_requeue" ]; then
    echo "[P3-ROLL-4N] $attempt in-allocation attempts exhausted; restart_count=$restarts/$max_requeue -> scontrol requeue for fresh nodes"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 10
    exit 0
  fi
  echo "[P3-ROLL-4N] giving up after $attempt in-place attempts and $restarts requeues" >&2
  exit "$rc"
done
