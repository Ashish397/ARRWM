#!/bin/bash
# One-node parameter arm for the decoder-shaped surrogate on the exact
# dmd10k_carncommit_long_off base (W&B g5ndc0fz).
#
# This launcher never submits or cancels Slurm jobs.  It occupies exactly one
# named node inside an existing two-node durable holder, allowing two matched
# arms to run concurrently without sharing GPUs.
set -euo pipefail

: "${HOLDER:?set HOLDER to the running two-node holder job id}"
: "${NODE:?set NODE to one node allocated to HOLDER}"
: "${PIXW:?set the calibrated decoder-surrogate generator weight}"
: "${TARGET_SHARE:?set the intended unweighted-gradient share label}"

PORTOFF=${PORTOFF:-28080}
RUNSTAMP=${RUNSTAMP:-$(date +%H%M%S)}
MAXSTEPS=${MAXSTEPS:-100}
SAMPLE_EVERY=${SAMPLE_EVERY:-15}
CHECKPOINT_EVERY=${CHECKPOINT_EVERY:-1000}
SAVE_FULL_CHECKPOINT=${SAVE_FULL_CHECKPOINT:-false}
KEEP_LAST_N_CHECKPOINTS=${KEEP_LAST_N_CHECKPOINTS:-4}
STATIONARY_FIRST7=${STATIONARY_FIRST7:-false}
HTF_REFERENCE_ROLLOUT=${HTF_REFERENCE_ROLLOUT:-false}
GAN_ENABLED=${GAN_ENABLED:-true}
GANLR=${GANLR:-2e-5}
GANUPDATES=${GANUPDATES:-1}
LADD_R1_GAMMA=${LADD_R1_GAMMA:-10}
LADD_DEFER_DISC_UPDATE=${LADD_DEFER_DISC_UPDATE:-false}
SURROGATE_FRESH_DISC_ORDER=${SURROGATE_FRESH_DISC_ORDER:-true}
GAN_ACTION_COND=${GAN_ACTION_COND:-false}
GAN_ACTION_CMAP_DIM=${GAN_ACTION_CMAP_DIM:-64}
GAN_ACTION_MISMATCH_WEIGHT=${GAN_ACTION_MISMATCH_WEIGHT:-0.0}
ACTION_CRITIC_FREEZE=${ACTION_CRITIC_FREEZE:-true}
ACTION_CRITIC_GUIDANCE_WEIGHT=${ACTION_CRITIC_GUIDANCE_WEIGHT:-0.3}
ACTION_TEACHER_MODE=${ACTION_TEACHER_MODE:-off}
ACTION_CRITIC_UPDATES=${ACTION_CRITIC_UPDATES:-1}
ACTION_CRITIC_Z_LOSS_WEIGHT=${ACTION_CRITIC_Z_LOSS_WEIGHT:-0.25}
ACTION_CRITIC_LR=${ACTION_CRITIC_LR:-3e-4}
TEACHER_ACTION_ENCODER=${TEACHER_ACTION_ENCODER:-pca_raw}
CARNMODE=${CARNMODE:-reference}
CARN_AUX_WEIGHT=${CARN_AUX_WEIGHT:-0.25}
CARN_COMMIT_ALPHA=${CARN_COMMIT_ALPHA:-0.25}
CARN_COMMIT_START_STEP=${CARN_COMMIT_START_STEP:-100}
CARN_COMMIT_RAMP_STEPS=${CARN_COMMIT_RAMP_STEPS:-100}
CARN_STAGED_CALIBRATION_APPROVED=${CARN_STAGED_CALIBRATION_APPROVED:-}
CARN_STAGED_CALIBRATION_RUN=${CARN_STAGED_CALIBRATION_RUN:-}
# Production brightness policy: expose raw RGB (including its evolving mean)
# to D, but project the generator-side pixel cotangent through the fixed B3
# high-pass/zero-mean contract.  ``none`` remains available only as an
# explicit historical ablation.
PIXEL_FILTER=${PIXEL_FILTER:-raw_grad_hp}
SWT_STRENGTH=${SWT_STRENGTH:-1.0}
STAT_ANCHOR=${STAT_ANCHOR:-1.0}
MEMORY_REDUCED_G=${MEMORY_REDUCED_G:-false}
EMPTY_CACHE_INTERVAL=${EMPTY_CACHE_INTERVAL:-20}
SURROGATE_STAGE=${SURROGATE_STAGE:-calibrate}
SRC=sbatch/carncommit_long.sbatch
BUNDLE=${DECODER_PULLBACK_BUNDLE:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256/exact_residual_ladder_2808/decoder_shaped_all_residual_seed0.pt}
EXPECTED_BUNDLE_SHA=de82ec3211ec408f34cca883be53c15d835e19d5a3860a0b7fe8840dad450357
EXPECTED_EXACT_STAGES=1,2,3,5,6,7,9,10,11,13,14,15
ARRWM_PYTHON=${ARRWM_PYTHON:-/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python3.10}

if ! [[ "$MAXSTEPS" =~ ^[1-9][0-9]*$ && "$SAMPLE_EVERY" =~ ^[0-9]+$ ]]; then
  echo "[CARN-SURROGATE] MAXSTEPS must be positive and SAMPLE_EVERY nonnegative" >&2
  exit 2
fi
if [ "$SAMPLE_EVERY" -gt 0 ] && [ "$SAMPLE_EVERY" -ge "$MAXSTEPS" ]; then
  echo "[CARN-SURROGATE] SAMPLE_EVERY must be below MAXSTEPS so videos are consumed" >&2
  exit 3
fi
if ! [[ "$EMPTY_CACHE_INTERVAL" =~ ^[1-9][0-9]*$ ]]; then
  echo "[CARN-SURROGATE] EMPTY_CACHE_INTERVAL must be positive" >&2
  exit 3
fi
if ! [[ "$CHECKPOINT_EVERY" =~ ^[1-9][0-9]*$ \
    && "$KEEP_LAST_N_CHECKPOINTS" =~ ^[1-9][0-9]*$ ]]; then
  echo "[CARN-SURROGATE] checkpoint interval/retention must be positive integers" >&2
  exit 3
fi
case "$SAVE_FULL_CHECKPOINT" in
  true|false) ;;
  *)
    echo "[CARN-SURROGATE] SAVE_FULL_CHECKPOINT must be true|false" >&2
    exit 3
    ;;
esac
case "$STATIONARY_FIRST7" in
  true|false) ;;
  *)
    echo "[CARN-SURROGATE] STATIONARY_FIRST7 must be true|false" >&2
    exit 3
    ;;
esac
case "$HTF_REFERENCE_ROLLOUT" in
  true|false) ;;
  *)
    echo "[CARN-SURROGATE] HTF_REFERENCE_ROLLOUT must be true|false" >&2
    exit 3
    ;;
esac
if [ "$STATIONARY_FIRST7" = "true" ] \
    && [ "$HTF_REFERENCE_ROLLOUT" = "true" ]; then
  echo "[CARN-SURROGATE] stationary and htf-reference rollout geometries are mutually exclusive" >&2
  exit 3
fi
case "$GAN_ENABLED" in
  true|false) ;;
  *)
    echo "[CARN-SURROGATE] GAN_ENABLED must be true|false" >&2
    exit 3
    ;;
esac
if ! [[ "$GANUPDATES" =~ ^[1-9][0-9]*$ ]]; then
  echo "[CARN-SURROGATE] GANUPDATES must be a positive integer; got $GANUPDATES" >&2
  exit 3
fi
case "$LADD_DEFER_DISC_UPDATE" in
  true|false) ;;
  *)
    echo "[CARN-SURROGATE] LADD_DEFER_DISC_UPDATE must be true|false; got $LADD_DEFER_DISC_UPDATE" >&2
    exit 3
    ;;
esac
case "$SURROGATE_FRESH_DISC_ORDER" in
  true|false) ;;
  *)
    echo "[CARN-SURROGATE] SURROGATE_FRESH_DISC_ORDER must be true|false; got $SURROGATE_FRESH_DISC_ORDER" >&2
    exit 3
    ;;
esac
case "$GAN_ACTION_COND" in
  true|false) ;;
  *)
    echo "[CARN-SURROGATE] GAN_ACTION_COND must be true|false; got $GAN_ACTION_COND" >&2
    exit 3
    ;;
esac
case "$ACTION_CRITIC_FREEZE" in
  true|false) ;;
  *)
    echo "[CARN-SURROGATE] ACTION_CRITIC_FREEZE must be true|false; got $ACTION_CRITIC_FREEZE" >&2
    exit 3
    ;;
esac
case "$ACTION_TEACHER_MODE" in
  off|slot0|all) ;;
  *)
    echo "[CARN-SURROGATE] ACTION_TEACHER_MODE must be off|slot0|all; got $ACTION_TEACHER_MODE" >&2
    exit 3
    ;;
esac
if [ "$ACTION_CRITIC_FREEZE" = "false" ] \
    && [ "$ACTION_TEACHER_MODE" = "off" ]; then
  echo "[CARN-SURROGATE] an online action critic requires ACTION_TEACHER_MODE=slot0|all" >&2
  exit 3
fi
if [ "$ACTION_CRITIC_FREEZE" = "true" ] \
    && [ "$ACTION_TEACHER_MODE" != "off" ]; then
  echo "[CARN-SURROGATE] a frozen action critic must keep ACTION_TEACHER_MODE=off to avoid an unused teacher" >&2
  exit 3
fi
if ! [[ "$ACTION_CRITIC_UPDATES" =~ ^[1-9][0-9]*$ ]]; then
  echo "[CARN-SURROGATE] ACTION_CRITIC_UPDATES must be a positive integer" >&2
  exit 3
fi
awk -v x="$ACTION_CRITIC_GUIDANCE_WEIGHT" 'BEGIN { exit !(x+0 >= 0) }' || {
  echo "[CARN-SURROGATE] ACTION_CRITIC_GUIDANCE_WEIGHT must be nonnegative" >&2
  exit 4
}
awk -v x="$ACTION_CRITIC_Z_LOSS_WEIGHT" 'BEGIN { exit !(x+0 > 0) }' || {
  echo "[CARN-SURROGATE] ACTION_CRITIC_Z_LOSS_WEIGHT must be positive" >&2
  exit 4
}
awk -v x="$ACTION_CRITIC_LR" 'BEGIN { exit !(x+0 > 0) }' || {
  echo "[CARN-SURROGATE] ACTION_CRITIC_LR must be positive" >&2
  exit 4
}
case "$TEACHER_ACTION_ENCODER" in
  ss_vae|pca_raw) ;;
  *)
    echo "[CARN-SURROGATE] TEACHER_ACTION_ENCODER must be ss_vae|pca_raw" >&2
    exit 4
    ;;
esac
case "$MEMORY_REDUCED_G" in
  true|false) ;;
  *)
    echo "[CARN-SURROGATE] MEMORY_REDUCED_G must be true|false; got $MEMORY_REDUCED_G" >&2
    exit 3
    ;;
esac
if [ "$SURROGATE_FRESH_DISC_ORDER" = "true" ] \
    && [ "$LADD_DEFER_DISC_UPDATE" != "false" ]; then
  echo "[CARN-SURROGATE] fresh discriminator order requires LADD_DEFER_DISC_UPDATE=false" >&2
  exit 3
fi
awk -v x="$PIXW" 'BEGIN { exit !(x+0 >= 0) }' || {
  echo "[CARN-SURROGATE] PIXW must be nonnegative; got $PIXW" >&2
  exit 4
}
awk -v x="$GANLR" 'BEGIN { exit !(x+0 > 0) }' || {
  echo "[CARN-SURROGATE] GANLR must be positive; got $GANLR" >&2
  exit 4
}
awk -v x="$LADD_R1_GAMMA" 'BEGIN { exit !(x+0 >= 0) }' || {
  echo "[CARN-SURROGATE] LADD_R1_GAMMA must be nonnegative; got $LADD_R1_GAMMA" >&2
  exit 4
}
if ! [[ "$GAN_ACTION_CMAP_DIM" =~ ^[1-9][0-9]*$ ]]; then
  echo "[CARN-SURROGATE] GAN_ACTION_CMAP_DIM must be positive" >&2
  exit 4
fi
awk -v x="$GAN_ACTION_MISMATCH_WEIGHT" 'BEGIN { exit !(x+0 >= 0) }' || {
  echo "[CARN-SURROGATE] GAN_ACTION_MISMATCH_WEIGHT must be nonnegative" >&2
  exit 4
}
if [ "$GAN_ACTION_COND" != "true" ]; then
  awk -v x="$GAN_ACTION_MISMATCH_WEIGHT" 'BEGIN { exit !(x+0 == 0) }' || {
    echo "[CARN-SURROGATE] a mismatch loss requires GAN_ACTION_COND=true" >&2
    exit 4
  }
fi
awk -v x="$CARN_AUX_WEIGHT" 'BEGIN { exit !(x+0 >= 0) }' || {
  echo "[CARN-SURROGATE] CARN_AUX_WEIGHT must be nonnegative; got $CARN_AUX_WEIGHT" >&2
  exit 4
}
awk -v x="$CARN_COMMIT_ALPHA" 'BEGIN { exit !(x+0 >= 0 && x+0 <= 1) }' || {
  echo "[CARN-SURROGATE] CARN_COMMIT_ALPHA must be in [0,1]; got $CARN_COMMIT_ALPHA" >&2
  exit 4
}
if ! [[ "$CARN_COMMIT_START_STEP" =~ ^[0-9]+$ \
    && "$CARN_COMMIT_RAMP_STEPS" =~ ^[0-9]+$ ]]; then
  echo "[CARN-SURROGATE] commit start/ramp steps must be nonnegative integers" >&2
  exit 4
fi
case "$PIXEL_FILTER" in
  none|dc|swt|dc_grad_hp|raw_grad_hp) ;;
  *)
    echo "[CARN-SURROGATE] PIXEL_FILTER must be none|dc|swt|dc_grad_hp|raw_grad_hp; got $PIXEL_FILTER" >&2
    exit 4
    ;;
esac
awk -v x="$SWT_STRENGTH" 'BEGIN { exit !(x+0 >= 0) }' || {
  echo "[CARN-SURROGATE] SWT_STRENGTH must be nonnegative; got $SWT_STRENGTH" >&2
  exit 4
}
awk -v x="$STAT_ANCHOR" 'BEGIN { exit !(x+0 >= 0) }' || {
  echo "[CARN-SURROGATE] STAT_ANCHOR must be nonnegative; got $STAT_ANCHOR" >&2
  exit 4
}
case "$SURROGATE_STAGE" in
  calibrate)
    awk -v x="$PIXW" 'BEGIN { exit !(x+0 == 0) }' || {
      echo "[CARN-SURROGATE] calibration requires PIXW=0; got $PIXW" >&2
      exit 4
    }
    ;;
  active)
    : "${SURROGATE_APPROVED:?set SURROGATE_APPROVED=YES after reviewing calibration}"
    : "${SURROGATE_CALIBRATION_RUN:?name the reviewed Q1~0.99 calibration run}"
    : "${SURROGATE_Q1_MIN:?set the reviewed strict Cartesian Q1}"
    [ "$SURROGATE_APPROVED" = "YES" ] || {
      echo "[CARN-SURROGATE] active stage requires SURROGATE_APPROVED=YES" >&2
      exit 4
    }
    awk -v x="$SURROGATE_Q1_MIN" 'BEGIN { exit !(x+0 >= 0.99) }' || {
      echo "[CARN-SURROGATE] strict Cartesian Q1 $SURROGATE_Q1_MIN is below 0.99" >&2
      exit 4
    }
    awk -v x="$PIXW" 'BEGIN { exit !(x+0 > 0) }' || {
      echo "[CARN-SURROGATE] active stage requires PIXW > 0; got $PIXW" >&2
      exit 4
    }
    ;;
  *)
    echo "[CARN-SURROGATE] SURROGATE_STAGE must be calibrate|active; got $SURROGATE_STAGE" >&2
    exit 4
    ;;
esac
[ -f "$BUNDLE" ] || {
  echo "[CARN-SURROGATE] missing decoder pullback bundle: $BUNDLE" >&2
  exit 5
}
ACTUAL_BUNDLE_SHA=$(sha256sum "$BUNDLE" | awk '{print $1}')
if [ "$ACTUAL_BUNDLE_SHA" != "$EXPECTED_BUNDLE_SHA" ]; then
  echo "[CARN-SURROGATE] Q1~0.99 bundle SHA mismatch: $ACTUAL_BUNDLE_SHA" >&2
  exit 5
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
  echo "[CARN-SURROGATE] obsolete/wrong exact residual stages: $BUNDLE_STAGES" >&2
  exit 5
fi
echo "[CARN-SURROGATE-BUNDLE] sha256=$ACTUAL_BUNDLE_SHA exact_residual_stages=[1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15] graph_free=true"

# Explicit CARN consumer policy.  `reference` is g5's commit-off cycle.
# `commit` changes only the proven committed-memory consumer.  `aux_minus`
# reproduces the completed 334wiw34 R2->R1 internalisation contract, and
# `commit_aux` is retained only as the historical immediate/full-strength
# ablation. `commit_aux_staged` is the promoted candidate: aux-minus stays at
# full strength while only the recurrent-memory consumer warms up and ramps.
# The harmony modes train distinct aligned F:R1->R2 and G:R2->R1 maps, then
# share G across aux-minus and staged commit.  The bridge adds the coherent
# one-step transition F(former)->G(F(latter)) under a fixed GAN loss budget.
# The two transition modes preserve the recent +former/-latter ablations
# without making either part of the reference calibration.
case "$CARNMODE" in
  reference)
    CARN_EXTRA="reverse_noiser_dedrift_apply_to_commit=false reverse_noiser_internalize_weight=0.0"
    ;;
  commit)
    CARN_EXTRA="reverse_noiser_dedrift_apply_to_commit=true reverse_noiser_internalize_weight=0.0"
    ;;
  aux_minus)
    CARN_EXTRA="forward_noiser_train_source=rollout forward_noiser_rollout2_source=legacy forward_noiser_reverse=true forward_noiser_cycle_enabled=false forward_noiser_apply_decoupled=false forward_noiser_apply_gt_former=false forward_noiser_apply_gt_both=false forward_noiser_apply_in_aux=false reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_apply_to_commit=false reverse_noiser_dedrift_apply_to_flash=false reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=$CARN_AUX_WEIGHT"
    ;;
  commit_aux)
    CARN_EXTRA="forward_noiser_train_source=rollout forward_noiser_rollout2_source=legacy forward_noiser_reverse=true forward_noiser_cycle_enabled=false forward_noiser_apply_decoupled=false forward_noiser_apply_gt_former=false forward_noiser_apply_gt_both=false forward_noiser_apply_in_aux=false reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_apply_to_commit=true reverse_noiser_dedrift_apply_to_flash=false reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=$CARN_AUX_WEIGHT"
    ;;
  commit_aux_staged)
    CARN_EXTRA="forward_noiser_enabled=true forward_noiser_train_source=rollout forward_noiser_rollout2_source=legacy forward_noiser_reverse=true forward_noiser_cycle_enabled=false forward_noiser_apply_decoupled=false forward_noiser_apply_gt_former=false forward_noiser_apply_gt_both=false forward_noiser_apply_in_aux=false reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_level=1 reverse_noiser_dedrift_min_level=1 reverse_noiser_dedrift_steps=1 reverse_noiser_dedrift_alpha0=1.0 reverse_noiser_internalize_weight=$CARN_AUX_WEIGHT reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_dedrift_apply_to_flash=false reverse_noiser_dedrift_apply_to_commit=true reverse_noiser_commit_alpha=$CARN_COMMIT_ALPHA reverse_noiser_commit_start_step=$CARN_COMMIT_START_STEP reverse_noiser_commit_ramp_steps=$CARN_COMMIT_RAMP_STEPS gen_aux_losses_x0_source=flash ladd_gt_transition_match=false stat_anchor_use_clean_match_offset=true stat_anchor_match_k=2 action_critic_aux_enabled=true action_critic_freeze=true generator_action_z_guidance_weight=0.3 ladd_pixel_swt_strength=0.0"
    ;;
  harmony_control)
    CARN_EXTRA="forward_noiser_enabled=true forward_noiser_train_source=rollout forward_noiser_rollout2_source=legacy forward_noiser_reverse=false forward_noiser_cycle_enabled=true cycle_rev_input_mode=paired_target cycle_freeze_g_in_cycle=true forward_noiser_pair_loss_weight=0.25 carn_recurse=false forward_noiser_apply_decoupled=false forward_noiser_apply_gt_former=false forward_noiser_apply_gt_both=false forward_noiser_apply_in_aux=false reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_level=1 reverse_noiser_dedrift_min_level=1 reverse_noiser_dedrift_steps=1 reverse_noiser_dedrift_alpha0=1.0 reverse_noiser_preserve_moments=true reverse_noiser_internalize_weight=$CARN_AUX_WEIGHT reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_dedrift_apply_to_flash=false reverse_noiser_dedrift_apply_to_commit=true reverse_noiser_commit_use_absolute_level=true reverse_noiser_commit_alpha=$CARN_COMMIT_ALPHA reverse_noiser_commit_start_step=$CARN_COMMIT_START_STEP reverse_noiser_commit_ramp_steps=$CARN_COMMIT_RAMP_STEPS gen_aux_losses_x0_source=flash ladd_gt_vs_fake_enabled=true ladd_gt_vs_fake_weight=1.0 ladd_gt_transition_enabled=false ladd_gt_transition_weight=0.0 ladd_gt_transition_carn_bridge=false ladd_gt_transition_carn_former=false ladd_gt_transition_carn_latter_reverse=false ladd_gt_transition_match=false stat_anchor_use_clean_match_offset=true stat_anchor_match_k=2 action_critic_aux_enabled=true action_critic_freeze=true generator_action_z_guidance_weight=0.3 ladd_pixel_swt_strength=0.0"
    ;;
  harmony_bridge)
    CARN_EXTRA="forward_noiser_enabled=true forward_noiser_train_source=rollout forward_noiser_rollout2_source=legacy forward_noiser_reverse=false forward_noiser_cycle_enabled=true cycle_rev_input_mode=paired_target cycle_freeze_g_in_cycle=true forward_noiser_pair_loss_weight=0.25 carn_recurse=false forward_noiser_apply_decoupled=false forward_noiser_apply_gt_former=false forward_noiser_apply_gt_both=false forward_noiser_apply_in_aux=false reverse_noiser_dedrift_enabled=true reverse_noiser_dedrift_level=1 reverse_noiser_dedrift_min_level=1 reverse_noiser_dedrift_steps=1 reverse_noiser_dedrift_alpha0=1.0 reverse_noiser_preserve_moments=true reverse_noiser_internalize_weight=$CARN_AUX_WEIGHT reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_dedrift_apply_to_flash=false reverse_noiser_dedrift_apply_to_commit=true reverse_noiser_commit_use_absolute_level=true reverse_noiser_commit_alpha=$CARN_COMMIT_ALPHA reverse_noiser_commit_start_step=$CARN_COMMIT_START_STEP reverse_noiser_commit_ramp_steps=$CARN_COMMIT_RAMP_STEPS gen_aux_losses_x0_source=flash ladd_gt_vs_fake_enabled=true ladd_gt_vs_fake_weight=0.5 ladd_gt_transition_enabled=true ladd_gt_transition_weight=0.5 ladd_gt_transition_carn_bridge=true ladd_gt_transition_carn_former=false ladd_gt_transition_carn_latter_reverse=false ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true ladd_gt_transition_match=false stat_anchor_use_clean_match_offset=true stat_anchor_match_k=2 action_critic_aux_enabled=true action_critic_freeze=true generator_action_z_guidance_weight=0.3 ladd_pixel_swt_strength=0.0"
    ;;
  former_plus)
    CARN_EXTRA="forward_noiser_train_source=flash forward_noiser_rollout2_source=legacy forward_noiser_reverse=false forward_noiser_cycle_enabled=false forward_noiser_apply_decoupled=false forward_noiser_apply_gt_former=false forward_noiser_apply_gt_both=false forward_noiser_apply_in_aux=false reverse_noiser_dedrift_enabled=false reverse_noiser_dedrift_apply_to_commit=false reverse_noiser_dedrift_apply_to_flash=false reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=0.0 ladd_gt_vs_fake_enabled=false ladd_gt_transition_enabled=true ladd_fake_backbone_grad_scale=0.2 ladd_gt_transition_carn_former=true ladd_gt_transition_carn_latter_reverse=false ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true"
    ;;
  latter_minus)
    CARN_EXTRA="forward_noiser_train_source=flash forward_noiser_rollout2_source=legacy forward_noiser_reverse=true forward_noiser_cycle_enabled=false forward_noiser_apply_decoupled=false forward_noiser_apply_gt_former=false forward_noiser_apply_gt_both=false forward_noiser_apply_in_aux=false reverse_noiser_dedrift_enabled=false reverse_noiser_dedrift_apply_to_commit=false reverse_noiser_dedrift_apply_to_flash=false reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=0.0 ladd_gt_vs_fake_enabled=false ladd_gt_transition_enabled=true ladd_fake_backbone_grad_scale=0.2 ladd_gt_transition_carn_former=false ladd_gt_transition_carn_latter_reverse=true ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true"
    ;;
  *)
    echo "[CARN-SURROGATE] CARNMODE must be reference|commit|aux_minus|commit_aux|commit_aux_staged|harmony_control|harmony_bridge|former_plus|latter_minus; got $CARNMODE" >&2
    exit 5
    ;;
esac

# Keep the inherited launcher's early provenance line consistent with the
# last-wins config injected below.  The resolved config remains authoritative,
# but a contradictory ``COMMITDEDRIFT=false`` line makes a healthy harmony
# run look half-enabled during log-only audits.
case "$CARNMODE" in
  commit|commit_aux|commit_aux_staged|harmony_control|harmony_bridge)
    COMMITDEDRIFT_ENV=true
    ;;
  *)
    COMMITDEDRIFT_ENV=false
    ;;
esac

# The historical staged recipe remains pinned to the DC-forward geometry on
# which it was calibrated.  Harmony may use either that control geometry or
# the promoted DC-off ``raw_grad_hp`` geometry; the latter preserves raw mean
# evolution while still rejecting broad generator-side brightness gradients.
if [ "$CARNMODE" = "commit_aux_staged" ]; then
  [ "$PIXEL_FILTER" = "dc" ] || {
    echo "[CARN-SURROGATE] commit_aux_staged requires its calibrated PIXEL_FILTER=dc" >&2
    exit 5
  }
elif [ "$CARNMODE" = "harmony_control" ] \
    || [ "$CARNMODE" = "harmony_bridge" ]; then
  case "$PIXEL_FILTER" in
    dc|raw_grad_hp) ;;
    *)
      echo "[CARN-SURROGATE] harmony modes require PIXEL_FILTER=dc|raw_grad_hp" >&2
      exit 5
      ;;
  esac
fi

if [ "$CARNMODE" = "commit_aux_staged" ] \
    || [ "$CARNMODE" = "harmony_control" ] \
    || [ "$CARNMODE" = "harmony_bridge" ]; then
  awk -v x="$STAT_ANCHOR" 'BEGIN { exit !(x+0 == 1) }' || {
    echo "[CARN-SURROGATE] staged/harmony modes require STAT_ANCHOR=1.0" >&2
    exit 5
  }
  awk -v x="$GANLR" 'BEGIN { exit !(x+0 == 1e-3) }' || {
    echo "[CARN-SURROGATE] staged/harmony modes require GANLR=1e-3" >&2
    exit 5
  }
  [ "$GANUPDATES" = "1" ] || {
    echo "[CARN-SURROGATE] staged/harmony modes require GANUPDATES=1" >&2
    exit 5
  }
  awk -v x="$LADD_R1_GAMMA" 'BEGIN { exit !(x+0 == 1) }' || {
    echo "[CARN-SURROGATE] staged/harmony modes require LADD_R1_GAMMA=1" >&2
    exit 5
  }
  [ "$LADD_DEFER_DISC_UPDATE" = "false" ] \
      && [ "$SURROGATE_FRESH_DISC_ORDER" = "true" ] || {
    echo "[CARN-SURROGATE] staged/harmony modes require cache-off fresh D-then-G order" >&2
    exit 5
  }
  if [ "$SURROGATE_STAGE" = "active" ]; then
    [ "$CARN_STAGED_CALIBRATION_APPROVED" = "YES" ] || {
      echo "[CARN-SURROGATE] active commit_aux_staged requires CARN_STAGED_CALIBRATION_APPROVED=YES" >&2
      exit 5
    }
    [ -n "$CARN_STAGED_CALIBRATION_RUN" ] || {
      echo "[CARN-SURROGATE] active commit_aux_staged requires its own weight-zero CARN_STAGED_CALIBRATION_RUN" >&2
      exit 5
    }
  fi
fi

MEMORY_EXTRA=""
MEMORY_TAG=""
if [ "$MEMORY_REDUCED_G" = "true" ]; then
  # Keep the full D evidence budget, but use the repository's proven G1x
  # geometry for the graph-free pullback. This halves the detached decoder
  # state bank and releases post-D storage before the field is captured.
  MEMORY_EXTRA="ladd_pixel_g_crops_per_row=1 ladd_pixel_g_lat_frames=2 ladd_pixel_g_frames_per_crop=2 ladd_pixel_g_decode_split=0 ladd_pixel_g_crop_stratify_x=0 ladd_pixel_post_d_memory_release=true"
  MEMORY_TAG="_g1x"
fi

GEOMETRY_EXTRA=""
GEOMETRY_TAG=""
if [ "$STATIONARY_FIRST7" = "true" ]; then
  # Phase-3 stationary curriculum: one fixed 21-frame/7-chunk AR rollout
  # from the ride start.  Do not slide the streaming window or activate the
  # phase-2 rolling-new/all-student-context branches.  Six generated chunks
  # follow the real seed chunk, giving the requested first-seven geometry.
  GEOMETRY_EXTRA="num_training_frames=21 rollout_frames=21 streaming_chunk_size=18 max_rolls_per_ride=1 dmd_only_first_chunk_per_ride=true dmd_42f_rolling_sup_new=false dmd_42f_allroll_student_ctx=false num_chunks_roll_forward=0 sample_7chunk_enabled=true max_ride_frames_random=false"
  GEOMETRY_TAG="_stationary7"
elif [ "$HTF_REFERENCE_ROLLOUT" = "true" ]; then
  # Exact native rollout construction used by W&B htfhu37j.  Keep these
  # values explicit/last-wins: pred_image_rollout is accumulated from each
  # ride's live generated ``new_frames`` and is emitted only after its depth
  # exceeds 21 latent frames.  This is not the independent 7+7 evaluator.
  GEOMETRY_EXTRA="num_training_frames=21 streaming_chunk_size=18 max_rolls_per_ride=4 dmd_only_first_chunk_per_ride=false dmd_42f_rolling_sup_new=true dmd_42f_allroll_student_ctx=true num_chunks_roll_forward=3 streaming_min_new_frame=9 streaming_max_length=60 sample_7chunk_enabled=true max_ride_frames_random=false rollout_viz_source=finish"
  GEOMETRY_TAG="_htfroll4"
fi

# Clean no-GAN ablation.  These are appended last so the inherited launcher
# cannot silently turn the discriminator or its decoder-pullback consumer back
# on.  Flash DMD itself remains enabled: the ablation removes only the
# adversarial objective/critic and its pixel-to-latent gradient path.
GAN_EXTRA=""
GAN_TAG=""
if [ "$GAN_ENABLED" = "false" ]; then
  GAN_EXTRA="gan_enabled=false gan_loss_weight=0.0 surrogate_decoder_shaped_enabled=false surrogate_decoder_fresh_disc_order=false pix_gan_weight=0.0 ladd_gt_vs_fake_enabled=false ladd_gt_transition_enabled=false ladd_adjacent_chunks_enabled=false gan_pixel_texture_enabled=false gan_of_enabled=false"
  GAN_TAG="_nogan"
fi

# Preserve the requested base's DMD/CARN/action-critic/Flash recipe.  The
# treatment replaces only its GAN feature/generator-gradient implementation:
# the same clean DMD-band GT-vs-fake discriminator is expressed in frozen VGG
# statistics, trained on the validated 5x evidence budget, and served to G by
# the Q1=.99 all-residual graph-free WAN pullback instead of an exact decoder
# backward. Every residual VJP is analytic/tied; only the stage-0 low core is
# learned.
#
# K=2,F=5,U=1 gives 10 logits/sample/step, equal to the base K=1,F=2,U=5.
# This branch's GAN is explicitly Flash-aligned.  D trains on the t=60 Flash
# slab and the Q1~0.99 pullback consumes a latest contiguous mask-live slice
# from that same slab.  `flash_dmd_enabled=true` alone would not establish
# either fact, hence the two independent source pins below.
COMMON="sample_interval=$SAMPLE_EVERY checkpoint_interval=$CHECKPOINT_EVERY keep_last_n_checkpoints=$KEEP_LAST_N_CHECKPOINTS auto_resume=false save_full_checkpoint=$SAVE_FULL_CHECKPOINT log_interval=5 memory_audit_enabled=true empty_cache_interval=$EMPTY_CACHE_INTERVAL gan_grad_telemetry_every=5 texture_tripwire_every=5 gan_pixel_texture_enabled=false gan_of_enabled=false fake_alt_head_enabled=false real_teacher_train_online=false dmd_frozen_teacher_pass_enabled=false boundary_vae_roundtrip=false state_probe_aux_enabled=false surrogate_critic_enabled=false surrogate_decoder_shaped_enabled=true surrogate_decoder_shaped_bundle=$BUNDLE surrogate_decoder_shaped_audit_every=0 surrogate_decoder_fresh_disc_order=$SURROGATE_FRESH_DISC_ORDER surrogate_teacher_backbone=ladd_pixel pix_gan_weight=$PIXW pix_r1_gamma=0.0 pix_crop_lat=[24,32] pix_lat_frames_per_crop=3 pix_finish_grad_enabled=false exit_exclude_last_rung=true pix_flash_grad_select_enabled=true ladd_feature_source=vgg ladd_fake_backbone_trainable=false ladd_defer_disc_update=$LADD_DEFER_DISC_UPDATE ladd_disc_loss_weight=0.0 ladd_r1_gamma=$LADD_R1_GAMMA ladd_use_prompt_cond=false ladd_cmap_dim=0 ladd_use_action_cond=$GAN_ACTION_COND ladd_action_cmap_dim=$GAN_ACTION_CMAP_DIM ladd_action_mismatch_weight=$GAN_ACTION_MISMATCH_WEIGHT ladd_disc_micro_batch_groups=8 ladd_pixel_crop_rows=24 ladd_pixel_crop_cols=32 ladd_pixel_crops_per_row=2 ladd_pixel_lat_frames=3 ladd_pixel_frames_per_crop=5 ladd_pixel_border_trim=8 ladd_pixel_decode_batch=4 ladd_pixel_decode_split=1 ladd_pixel_decode_cache=false ladd_pixel_crop_stratify_x=1 ladd_pixel_crop_y_lo_frac=0.0 ladd_pixel_crop_y_hi_frac=1.0 ladd_pixel_encoder_trainable=false ladd_pixel_encoder_lr_scale=0.0 ladd_pixel_input_filter=$PIXEL_FILTER ladd_pixel_swt_strength=$SWT_STRENGTH stat_anchor_loss_weight=$STAT_ANCHOR gan_lr=$GANLR gan_updates_per_step=$GANUPDATES ladd_fake_sample_source=flash ladd_disc_force_clean=true flash_dmd_enabled=true flash_dmd_gan_t=60 $CARN_EXTRA action_critic_aux_enabled=true action_critic_freeze=$ACTION_CRITIC_FREEZE action_teacher_mode=$ACTION_TEACHER_MODE teacher_action_encoder=$TEACHER_ACTION_ENCODER generator_action_z_guidance_weight=$ACTION_CRITIC_GUIDANCE_WEIGHT critic_updates_per_step=$ACTION_CRITIC_UPDATES action_critic_z_loss_weight=$ACTION_CRITIC_Z_LOSS_WEIGHT critic_lr=$ACTION_CRITIC_LR $MEMORY_EXTRA $GEOMETRY_EXTRA $GAN_EXTRA"

PIXW_TAG=${PIXW//./p}
SHARE_TAG=${TARGET_SHARE//./p}
GANLR_TAG=${GANLR//./p}
GANLR_TAG=${GANLR_TAG//-/m}
R1_TAG=${LADD_R1_GAMMA//./p}
if [ "$GAN_ACTION_COND" = "true" ]; then
  ACTION_TAG="_actcond_mis${GAN_ACTION_MISMATCH_WEIGHT//./p}"
else
  ACTION_TAG="_actblind"
fi
if [ "$ACTION_CRITIC_FREEZE" = "true" ]; then
  CRITIC_TAG="acfrozen_g${ACTION_CRITIC_GUIDANCE_WEIGHT//./p}"
else
  CRITIC_TAG="aconline_${ACTION_TEACHER_MODE}_u${ACTION_CRITIC_UPDATES}_g${ACTION_CRITIC_GUIDANCE_WEIGHT//./p}"
fi
if [ "$SURROGATE_FRESH_DISC_ORDER" = "true" ]; then
  DORDER_TAG=dfresh_DthenG
elif [ "$LADD_DEFER_DISC_UPDATE" = "true" ]; then
  DORDER_TAG=ddefer_GthenD
else
  DORDER_TAG=dinline_GthenD
fi
if [ "$GAN_ENABLED" = "false" ]; then
  DARM="carnbaseoff_flash60_nogan_${CARNMODE}_share${SHARE_TAG}_${PIXEL_FILTER}_sa${STAT_ANCHOR}${ACTION_TAG}_${CRITIC_TAG}${MEMORY_TAG}${GEOMETRY_TAG}_${RUNSTAMP}"
else
  DARM="carnbaseoff_flashgan60_decoderpullback_q1p99_${SURROGATE_STAGE}_${CARNMODE}_share${SHARE_TAG}_${PIXEL_FILTER}_sa${STAT_ANCHOR}_w${PIXW_TAG}_dlr${GANLR_TAG}_du${GANUPDATES}_r1${R1_TAG}${ACTION_TAG}_${CRITIC_TAG}_${DORDER_TAG}${MEMORY_TAG}${GEOMETRY_TAG}_${RUNSTAMP}"
fi

echo "[CARN-SURROGATE-CONFIG] source=$SRC base_wandb=g5ndc0fz"
echo "[CARN-SURROGATE-CONFIG] holder=$HOLDER node=$NODE stage=$SURROGATE_STAGE share=$TARGET_SHARE gan_enabled=$GAN_ENABLED pixw=$PIXW gan_lr=$GANLR gan_updates=$GANUPDATES ladd_r1_gamma=$LADD_R1_GAMMA action_cond=$GAN_ACTION_COND action_cmap_dim=$GAN_ACTION_CMAP_DIM action_mismatch_weight=$GAN_ACTION_MISMATCH_WEIGHT action_critic_freeze=$ACTION_CRITIC_FREEZE action_critic_guidance=$ACTION_CRITIC_GUIDANCE_WEIGHT action_teacher_mode=$ACTION_TEACHER_MODE action_critic_updates=$ACTION_CRITIC_UPDATES action_critic_z_weight=$ACTION_CRITIC_Z_LOSS_WEIGHT action_critic_lr=$ACTION_CRITIC_LR teacher_action_encoder=$TEACHER_ACTION_ENCODER defer_disc_update=$LADD_DEFER_DISC_UPDATE fresh_disc_order=$SURROGATE_FRESH_DISC_ORDER carn_mode=$CARNMODE carn_aux_weight=$CARN_AUX_WEIGHT commit_alpha=$CARN_COMMIT_ALPHA commit_start=$CARN_COMMIT_START_STEP commit_ramp=$CARN_COMMIT_RAMP_STEPS pixel_filter=$PIXEL_FILTER swt_strength=$SWT_STRENGTH stat_anchor=$STAT_ANCHOR stationary_first7=$STATIONARY_FIRST7 htf_reference_rollout=$HTF_REFERENCE_ROLLOUT checkpoint_every=$CHECKPOINT_EVERY save_full_checkpoint=$SAVE_FULL_CHECKPOINT keep_last=$KEEP_LAST_N_CHECKPOINTS memory_reduced_g=$MEMORY_REDUCED_G empty_cache_interval=$EMPTY_CACHE_INTERVAL"
if [ "$CARNMODE" = "commit_aux_staged" ]; then
  echo "[CARN-SURROGATE-CONFIG] staged_calibration_run=${CARN_STAGED_CALIBRATION_RUN:-pending} staged_calibration_approved=${CARN_STAGED_CALIBRATION_APPROVED:-NO}"
fi
echo "[CARN-SURROGATE-CONFIG] run_name=dmd10k_${DARM}_j${HOLDER}"
echo "[CARN-SURROGATE-CONFIG] max_steps=$MAXSTEPS last_wins=[$COMMON]"

if [ "${PRINT_ONLY:-0}" = "1" ]; then
  exit 0
fi

NODE_EXPR=$(squeue -j "$HOLDER" -h -o %N)
if [ -z "$NODE_EXPR" ] || [ "$NODE_EXPR" = "(null)" ]; then
  echo "[CARN-SURROGATE] holder $HOLDER is not running" >&2
  exit 6
fi
if ! scontrol show hostnames "$NODE_EXPR" | grep -Fxq "$NODE"; then
  echo "[CARN-SURROGATE] node $NODE is not allocated to holder $HOLDER ($NODE_EXPR)" >&2
  exit 7
fi

export COMMON DARM MAXSTEPS
TMP0=$(mktemp "/tmp/carnsur_${HOLDER}_${NODE}_${PORTOFF}_inject.XXXXXX")
TMP1=$(mktemp "/tmp/carnsur_${HOLDER}_${NODE}_${PORTOFF}_run.XXXXXX")
trap 'rm -f "$TMP0" "$TMP1"' EXIT

N_RUN_NAME=$(grep -c '^[[:space:]]*run_name=dmd10k_' "$SRC")
if [ "$N_RUN_NAME" -ne 1 ]; then
  echo "[CARN-SURROGATE] expected one run_name injection site; got $N_RUN_NAME" >&2
  exit 8
fi
awk '
  /^[[:space:]]*run_name=dmd10k_/ { print "    " ENVIRON["COMMON"] " \\" }
  { print }
' "$SRC" > "$TMP0"

MASTER_PORT=$((29500 + (HOLDER + PORTOFF) % 16000))
sed -e 's/^srun torchrun \\/srun --jobid='"$HOLDER"' --overlap --nodelist='"$NODE"' --nodes=1 --ntasks=1 --gpus-per-node=4 --gpu-bind=none torchrun \\/' \
    -e 's/--nnodes=\$SLURM_NNODES/--nnodes=1/' \
    -e 's/--rdzv_id=\$SLURM_JOB_ID/--rdzv_id='"$HOLDER$PORTOFF"'/' \
    -e 's|--rdzv_endpoint=\${MASTER_ADDR}:\${MASTER_PORT}|--rdzv_endpoint='"$NODE:$MASTER_PORT"'|' \
    -e 's/^#SBATCH.*//' \
    "$TMP0" > "$TMP1"

LOG="logs/carn_surrogate_q1p99_share${SHARE_TAG}_h${HOLDER}_${NODE}_${RUNSTAMP}.log"
{
  echo "[CARN-SURROGATE-CONFIG] source=$SRC base_wandb=g5ndc0fz"
  echo "[CARN-SURROGATE-CONFIG] holder=$HOLDER node=$NODE stage=$SURROGATE_STAGE share=$TARGET_SHARE pixw=$PIXW gan_lr=$GANLR gan_updates=$GANUPDATES action_cond=$GAN_ACTION_COND action_cmap_dim=$GAN_ACTION_CMAP_DIM action_mismatch_weight=$GAN_ACTION_MISMATCH_WEIGHT action_critic_freeze=$ACTION_CRITIC_FREEZE action_critic_guidance=$ACTION_CRITIC_GUIDANCE_WEIGHT action_teacher_mode=$ACTION_TEACHER_MODE action_critic_updates=$ACTION_CRITIC_UPDATES action_critic_z_weight=$ACTION_CRITIC_Z_LOSS_WEIGHT action_critic_lr=$ACTION_CRITIC_LR teacher_action_encoder=$TEACHER_ACTION_ENCODER defer_disc_update=$LADD_DEFER_DISC_UPDATE fresh_disc_order=$SURROGATE_FRESH_DISC_ORDER carn_mode=$CARNMODE carn_aux_weight=$CARN_AUX_WEIGHT commit_alpha=$CARN_COMMIT_ALPHA commit_start=$CARN_COMMIT_START_STEP commit_ramp=$CARN_COMMIT_RAMP_STEPS pixel_filter=$PIXEL_FILTER swt_strength=$SWT_STRENGTH stat_anchor=$STAT_ANCHOR memory_reduced_g=$MEMORY_REDUCED_G empty_cache_interval=$EMPTY_CACHE_INTERVAL"
  if [ "$CARNMODE" = "commit_aux_staged" ]; then
    echo "[CARN-SURROGATE-CONFIG] staged_calibration_run=${CARN_STAGED_CALIBRATION_RUN:-pending} staged_calibration_approved=${CARN_STAGED_CALIBRATION_APPROVED:-NO}"
  fi
  echo "[CARN-SURROGATE-CONFIG] run_name=dmd10k_${DARM}_j${HOLDER}"
  echo "[CARN-SURROGATE-CONFIG] max_steps=$MAXSTEPS last_wins=[$COMMON]"
} > "$LOG"

set +e
SLURM_JOB_ID=$HOLDER SLURM_NNODES=1 SLURM_JOB_NODELIST=$NODE \
  COMMITDEDRIFT=$COMMITDEDRIFT_ENV DARM=$DARM MAXSTEPS=$MAXSTEPS TELEM=5 \
  bash "$TMP1" >> "$LOG" 2>&1
RC=$?
set -e
echo "[CARN-SURROGATE] exit=$RC log=$LOG"
exit "$RC"
