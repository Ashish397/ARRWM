#!/bin/bash
# Definitive stationary Phase-3 treatment and single-axis ablations at
# 8 nodes x 4 GPUs/node x one ride/rank = global batch size 32.
#
# Submit with exactly one PHASE3_AXIS:
#   base | base_v2 | no_gan | no_commit | no_aux | no_action_critic | no_stat_anchor
#
# The existing two-node source is executed inside this allocation after a
# final last-wins override block is injected immediately before run_name.
# Its #SBATCH lines are comments when invoked through bash, while its native
# `srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=4` consumes all eight
# nodes in this allocation.
#SBATCH --job-name=p3def8n
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -euo pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

: "${PHASE3_AXIS:?set PHASE3_AXIS=base|base_v2|no_gan|no_commit|no_aux|no_action_critic|no_stat_anchor}"

case "$PHASE3_AXIS" in
  base|base_v2|no_gan|no_commit|no_aux|no_action_critic|no_stat_anchor) ;;
  *)
    echo "[P3-8N] invalid PHASE3_AXIS=$PHASE3_AXIS" >&2
    exit 2
    ;;
esac

EXPECTED_NODES="${PHASE3_EXPECTED_NODES:-8}"
if [ "${PHASE3_PRINT_ONLY:-0}" != "1" ]; then
  if [ "${SLURM_NNODES:-0}" -ne "$EXPECTED_NODES" ]; then
    echo "[P3-8N] expected exactly $EXPECTED_NODES nodes; got ${SLURM_NNODES:-unset}" >&2
    exit 3
  fi
fi

SRC=sbatch/carncommit_long.sbatch
BUNDLE=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256/exact_residual_ladder_2808/decoder_shaped_all_residual_seed0.pt
EXPECTED_BUNDLE_SHA=de82ec3211ec408f34cca883be53c15d835e19d5a3860a0b7fe8840dad450357
EXPECTED_EXACT_STAGES=1,2,3,5,6,7,9,10,11,13,14,15
ARRWM_PYTHON=/scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python3.10

[ -f "$SRC" ] || {
  echo "[P3-8N] missing source launcher: $SRC" >&2
  exit 4
}
[ -f "$BUNDLE" ] || {
  echo "[P3-8N] missing Q1~0.99 bundle: $BUNDLE" >&2
  exit 4
}

ACTUAL_BUNDLE_SHA=$(sha256sum "$BUNDLE" | awk '{print $1}')
if [ "$ACTUAL_BUNDLE_SHA" != "$EXPECTED_BUNDLE_SHA" ]; then
  echo "[P3-8N] Q1~0.99 bundle SHA mismatch: $ACTUAL_BUNDLE_SHA" >&2
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
  echo "[P3-8N] wrong exact residual stages: $BUNDLE_STAGES" >&2
  exit 4
fi

# Exact definitive treatment. This block comes after every historical setting
# in carncommit_long.sbatch, so it is the resolved last-wins authority.
BASE_COMMON="max_steps=200 sample_interval=15 checkpoint_interval=100 keep_last_n_checkpoints=4 auto_resume=false save_full_checkpoint=true log_interval=5 memory_audit_enabled=true empty_cache_interval=20 gan_grad_telemetry_every=5 texture_tripwire_every=5 gan_pixel_texture_enabled=false gan_of_enabled=false fake_alt_head_enabled=false real_teacher_train_online=false dmd_frozen_teacher_pass_enabled=false real_teacher_causal_mask=false fake_score_causal_mask=false boundary_vae_roundtrip=false state_probe_aux_enabled=false surrogate_critic_enabled=false surrogate_decoder_shaped_enabled=true surrogate_decoder_shaped_bundle=$BUNDLE surrogate_decoder_shaped_audit_every=0 surrogate_decoder_fresh_disc_order=true surrogate_teacher_backbone=ladd_pixel pix_gan_weight=0.0175 pix_r1_gamma=0.0 pix_crop_lat=[24,32] pix_lat_frames_per_crop=3 pix_finish_grad_enabled=false exit_exclude_last_rung=true pix_flash_grad_select_enabled=true ladd_feature_source=vgg ladd_fake_backbone_trainable=false ladd_defer_disc_update=false ladd_disc_loss_weight=0.0 ladd_r1_gamma=1 ladd_use_prompt_cond=false ladd_cmap_dim=0 ladd_use_action_cond=false ladd_action_cmap_dim=64 ladd_action_mismatch_weight=0.0 ladd_disc_micro_batch_groups=8 ladd_pixel_crop_rows=24 ladd_pixel_crop_cols=32 ladd_pixel_crops_per_row=2 ladd_pixel_lat_frames=3 ladd_pixel_frames_per_crop=5 ladd_pixel_border_trim=8 ladd_pixel_decode_batch=4 ladd_pixel_decode_split=1 ladd_pixel_decode_cache=false ladd_pixel_crop_stratify_x=1 ladd_pixel_crop_y_lo_frac=0.0 ladd_pixel_crop_y_hi_frac=1.0 ladd_pixel_encoder_trainable=false ladd_pixel_encoder_lr_scale=0.0 ladd_pixel_input_filter=raw_grad_hp ladd_pixel_swt_strength=0.0 stat_anchor_loss_weight=1.0 gan_lr=1e-3 gan_updates_per_step=1 ladd_fake_sample_source=flash ladd_disc_force_clean=true flash_dmd_enabled=true flash_dmd_gan_t=60"

# Coherent harmony bridge: paired F:R1->R2 and G:R2->R1, aux-minus 0.25,
# and committed-memory correction staged 0 -> 0.25 over steps 100..200.
CARN_COMMON="forward_noiser_enabled=true forward_noiser_train_source=rollout forward_noiser_rollout2_source=legacy forward_noiser_reverse=false forward_noiser_cycle_enabled=true cycle_rev_input_mode=paired_target cycle_freeze_g_in_cycle=true forward_noiser_pair_loss_weight=0.25 carn_recurse=false forward_noiser_apply_decoupled=false forward_noiser_apply_gt_former=false forward_noiser_apply_gt_both=false forward_noiser_apply_in_aux=false reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_level=1 reverse_noiser_dedrift_min_level=1 reverse_noiser_dedrift_steps=1 reverse_noiser_dedrift_alpha0=1.0 reverse_noiser_preserve_moments=true reverse_noiser_internalize_weight=0.25 reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_dedrift_apply_to_flash=false reverse_noiser_dedrift_apply_to_commit=true reverse_noiser_commit_use_absolute_level=true reverse_noiser_commit_alpha=0.25 reverse_noiser_commit_start_step=100 reverse_noiser_commit_ramp_steps=100 gen_aux_losses_x0_source=flash ladd_gt_vs_fake_enabled=true ladd_gt_vs_fake_weight=0.5 ladd_gt_transition_enabled=true ladd_gt_transition_weight=0.5 ladd_gt_transition_carn_bridge=true ladd_gt_transition_carn_former=false ladd_gt_transition_carn_latter_reverse=false ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true ladd_gt_transition_match=false stat_anchor_use_clean_match_offset=false stat_anchor_match_k=2 action_critic_aux_enabled=true action_critic_freeze=true action_teacher_mode=off teacher_action_encoder=pca_raw generator_action_z_guidance_weight=0.3 lora_action_critic_z_guidance_weight=0.0 critic_updates_per_step=1 action_critic_z_loss_weight=0.25 critic_lr=3e-4"

# Generate a full seven-chunk student window and retain the graph for every
# block. DMD and the non-DMD CARN/GAN auxiliaries supervise S1..S7 (21 frames).
# The clean side is positional GT only: no temporal clean-window matching.
GEOMETRY_COMMON="num_training_frames=21 rollout_frames=21 streaming_chunk_size=21 max_rolls_per_ride=1 dmd_only_first_chunk_per_ride=true dmd_42f_enabled=true dmd_42f_num_chunks=7 dmd_42f_seed_last=false dmd_42f_gt_after_chunks=1 dmd_42f_gt_anchor=true dmd_42f_rand_sup_slot=false dmd_42f_allsup=false dmd_42f_rolling_sup_new=false dmd_42f_allroll_student_ctx=false dmd_42f_clean_match_enabled=false dmd_42f_clean_match_forward=false dmd_42f_clean_match_forward_only=false aux_train_trailing_chunk=true dmd_42f_supervise_last=true num_chunks_roll_forward=0 sample_7chunk_enabled=true max_ride_frames_random=false"

ABLATION_EXTRA=""
COMMITDEDRIFT_ENV=true
case "$PHASE3_AXIS" in
  base)
    ;;
  base_v2)
    # Faithful stationary parent for the rehabilitated rolling base. Retire
    # every transition-GAN route, give the full pair-mode budget to GT-vs-fake,
    # and align both GAN and auxiliary guidance to the bounded post-CARN commit
    # endpoint. This matches the current rolling base_v2 loss surface from the
    # first stationary step instead of inheriting the historical transition
    # bridge checkpoint.
    ABLATION_EXTRA="pix_finish_grad_enabled=true ladd_fake_sample_source=commit reverse_noiser_commit_alpha=0.125 reverse_noiser_commit_start_step=150 reverse_noiser_commit_ramp_steps=100 reverse_noiser_commit_max_relative_shift=0.015 gan_carn_commit_align_enabled=true gan_carn_commit_straight_through=true gen_aux_losses_x0_source=commit ladd_gt_vs_fake_enabled=true ladd_gt_vs_fake_weight=1.0 ladd_gt_transition_enabled=false ladd_gt_transition_weight=0.0 ladd_gt_transition_carn_bridge=false ladd_gt_transition_carn_former=false ladd_gt_transition_carn_latter_reverse=false ladd_gt_transition_gen_detach_former=false ladd_gt_transition_match=false ladd_adjacent_chunks_enabled=false"
    ;;
  no_gan)
    # Remove the discriminator, decoder pullback, and every LADD pair loss.
    # Flash DMD, CARN cycle/aux/commit, action critic, and stat anchor remain.
    ABLATION_EXTRA="gan_enabled=false gan_loss_weight=0.0 surrogate_decoder_shaped_enabled=false surrogate_decoder_fresh_disc_order=false pix_gan_weight=0.0 ladd_gt_vs_fake_enabled=false ladd_gt_transition_enabled=false ladd_adjacent_chunks_enabled=false gan_pixel_texture_enabled=false gan_of_enabled=false"
    ;;
  no_commit)
    # Keep paired F/G, harmony and aux-minus; remove only recurrent correction.
    ABLATION_EXTRA="reverse_noiser_dedrift_apply_to_commit=false reverse_noiser_commit_alpha=0.0"
    COMMITDEDRIFT_ENV=false
    ;;
  no_aux)
    # Keep paired F/G, harmony and staged recurrent correction; remove only the
    # generator internalisation loss sourced from G(raw rollout).
    ABLATION_EXTRA="reverse_noiser_internalize_weight=0.0"
    ;;
  no_action_critic)
    # Preserve ordinary generator action conditioning; remove only the frozen
    # auxiliary action critic's generator guidance route. The critic checkpoint
    # path is also blanked so the loader skips its strict load: this dotlist is
    # appended last and wins over the action_critic_checkpoint= set by
    # carncommit_long.sbatch, which otherwise raises RuntimeError when the head
    # is not instantiated (job 6176584 died at startup on exactly that).
    ABLATION_EXTRA="action_critic_aux_enabled=false action_teacher_mode=off generator_action_z_guidance_weight=0.0 lora_action_critic_z_guidance_weight=0.0 action_critic_checkpoint="
    ;;
  no_stat_anchor)
    # Keep the per-stat recipe present but make its outer contribution zero.
    ABLATION_EXTRA="stat_anchor_loss_weight=0.0"
    ;;
esac

SMOKE_EXTRA=""
if [ "${PHASE3_SMOKE:-0}" = "1" ]; then
  # Holder-only GPU path check. Keep the full model/loss/geometry recipe, but
  # default to one optimizer step without W&B, sampling, or checkpoint writes.
  # Explicit smoke overrides allow longer observable holder validation without
  # changing any production default captured by the queued p3d8 jobs.
  SMOKE_MAX_STEPS="${PHASE3_SMOKE_MAX_STEPS:-1}"
  SMOKE_SAMPLE_INTERVAL="${PHASE3_SMOKE_SAMPLE_INTERVAL:-0}"
  SMOKE_DISABLE_WANDB="${PHASE3_SMOKE_DISABLE_WANDB:-true}"
  SMOKE_SAMPLE_7CHUNK="${PHASE3_SMOKE_SAMPLE_7CHUNK:-false}"
  SMOKE_RECIPE_EXTRA="${PHASE3_SMOKE_RECIPE_EXTRA:-}"
  SMOKE_EXTRA="max_steps=$SMOKE_MAX_STEPS sample_interval=$SMOKE_SAMPLE_INTERVAL checkpoint_interval=999999 keep_last_n_checkpoints=1 auto_resume=false save_full_checkpoint=false disable_wandb=$SMOKE_DISABLE_WANDB log_interval=1 dmd_loss_warmup_steps=0 sample_7chunk_enabled=$SMOKE_SAMPLE_7CHUNK $SMOKE_RECIPE_EXTRA"
fi

COMMON="$BASE_COMMON $CARN_COMMON $GEOMETRY_COMMON $ABLATION_EXTRA $SMOKE_EXTRA"
export COMMON

DARM="${PHASE3_DARM_OVERRIDE:-phase3_stationary_definitive_8n_${PHASE3_AXIS}_gbs32_2808}"
export DARM
if [ "${PHASE3_SMOKE:-0}" = "1" ]; then
  MAXSTEPS="$SMOKE_MAX_STEPS"
else
  MAXSTEPS=200
fi
export MAXSTEPS

N_RUN_NAME=$(grep -c '^[[:space:]]*run_name=dmd10k_' "$SRC")
if [ "$N_RUN_NAME" -ne 1 ]; then
  echo "[P3-8N] expected one run_name injection site; got $N_RUN_NAME" >&2
  exit 5
fi

TMP_LAUNCH=$(mktemp "/tmp/p3def8n_${PHASE3_AXIS}_${SLURM_JOB_ID:-print}.XXXXXX")
trap 'rm -f "$TMP_LAUNCH"' EXIT
awk '
  /^[[:space:]]*run_name=dmd10k_/ { print "    " ENVIRON["COMMON"] " \\" }
  { print }
' "$SRC" > "$TMP_LAUNCH"
bash -n "$TMP_LAUNCH"

echo "[P3-8N] axis=$PHASE3_AXIS nodes=${SLURM_NNODES:-$EXPECTED_NODES} ranks_per_node=4 global_batch=$((EXPECTED_NODES * 4)) max_steps=$MAXSTEPS time_limit=03:00:00 smoke=${PHASE3_SMOKE:-0}"
echo "[P3-8N] samples=15 checkpoint_interval=100 save_full_checkpoint=true (production defaults; smoke overrides are last-wins)"
echo "[P3-8N] bundle_sha=$ACTUAL_BUNDLE_SHA exact_residual_stages=[$BUNDLE_STAGES]"
echo "[P3-8N] DARM=$DARM"
echo "[P3-8N] last_wins=[$COMMON]"

if [ "${PHASE3_PRINT_ONLY:-0}" = "1" ]; then
  exit 0
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
    echo "[P3-8N] attempt $attempt failed rc=$rc after ${launch_dt}s -> in-allocation relaunch with fresh rdzv port"
    srun --ntasks="$SLURM_NNODES" --ntasks-per-node=1 bash -c 'pkill -9 -f "[c]ausal_action_forcing_train.py" 2>/dev/null; sleep 3; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader 2>/dev/null | sed "s/^/[hbm-drain $(hostname)] /"' || true
    sleep 45
    continue
  fi
  if [ "$launch_dt" -ge "$inplace_window" ]; then
    echo "[P3-8N] attempt $attempt failed rc=$rc after ${launch_dt}s (>= ${inplace_window}s window): not a launch race, giving up" >&2
    exit "$rc"
  fi
  restarts="${SLURM_RESTART_COUNT:-0}"
  max_requeue="${PHASE3_MAX_REQUEUE:-4}"
  if [ -n "${SLURM_JOB_ID:-}" ] && [ "$restarts" -lt "$max_requeue" ]; then
    echo "[P3-8N] $attempt in-allocation attempts exhausted; restart_count=$restarts/$max_requeue -> scontrol requeue for fresh nodes"
    scontrol requeue "$SLURM_JOB_ID"
    sleep 10
    exit 0
  fi
  echo "[P3-8N] giving up after $attempt in-place attempts and $restarts requeues" >&2
  exit "$rc"
done
