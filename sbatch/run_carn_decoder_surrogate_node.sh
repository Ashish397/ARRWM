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
GANLR=${GANLR:-2e-5}
GANUPDATES=${GANUPDATES:-1}
LADD_R1_GAMMA=${LADD_R1_GAMMA:-10}
LADD_DEFER_DISC_UPDATE=${LADD_DEFER_DISC_UPDATE:-false}
CARNMODE=${CARNMODE:-reference}
CARN_AUX_WEIGHT=${CARN_AUX_WEIGHT:-0.25}
PIXEL_FILTER=${PIXEL_FILTER:-none}
SWT_STRENGTH=${SWT_STRENGTH:-1.0}
STAT_ANCHOR=${STAT_ANCHOR:-1.0}
SURROGATE_STAGE=${SURROGATE_STAGE:-calibrate}
SRC=sbatch/carncommit_long.sbatch
BUNDLE=${DECODER_PULLBACK_BUNDLE:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256/exact_residual_ladder_2808/decoder_shaped_all_residual_seed0.pt}
EXPECTED_BUNDLE_SHA=de82ec3211ec408f34cca883be53c15d835e19d5a3860a0b7fe8840dad450357
EXPECTED_EXACT_STAGES=1,2,3,5,6,7,9,10,11,13,14,15
ARRWM_PYTHON=${ARRWM_PYTHON:-/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/envs/arrwm/bin/python3.10}

if ! [[ "$MAXSTEPS" =~ ^[1-9][0-9]*$ && "$SAMPLE_EVERY" =~ ^[1-9][0-9]*$ ]]; then
  echo "[CARN-SURROGATE] MAXSTEPS and SAMPLE_EVERY must be positive integers" >&2
  exit 2
fi
if [ "$SAMPLE_EVERY" -ge "$MAXSTEPS" ]; then
  echo "[CARN-SURROGATE] SAMPLE_EVERY must be below MAXSTEPS so videos are consumed" >&2
  exit 3
fi
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
awk -v x="$CARN_AUX_WEIGHT" 'BEGIN { exit !(x+0 >= 0) }' || {
  echo "[CARN-SURROGATE] CARN_AUX_WEIGHT must be nonnegative; got $CARN_AUX_WEIGHT" >&2
  exit 4
}
case "$PIXEL_FILTER" in
  none|dc|swt) ;;
  *)
    echo "[CARN-SURROGATE] PIXEL_FILTER must be none|dc|swt; got $PIXEL_FILTER" >&2
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
# `commit_aux` combines those two already-tested consumers.  The two
# transition modes preserve the recent +former/-latter ablations without
# making either part of the reference calibration.
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
  former_plus)
    CARN_EXTRA="forward_noiser_train_source=flash forward_noiser_rollout2_source=legacy forward_noiser_reverse=false forward_noiser_cycle_enabled=false forward_noiser_apply_decoupled=false forward_noiser_apply_gt_former=false forward_noiser_apply_gt_both=false forward_noiser_apply_in_aux=false reverse_noiser_dedrift_enabled=false reverse_noiser_dedrift_apply_to_commit=false reverse_noiser_dedrift_apply_to_flash=false reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=0.0 ladd_gt_vs_fake_enabled=false ladd_gt_transition_enabled=true ladd_fake_backbone_grad_scale=0.2 ladd_gt_transition_carn_former=true ladd_gt_transition_carn_latter_reverse=false ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true"
    ;;
  latter_minus)
    CARN_EXTRA="forward_noiser_train_source=flash forward_noiser_rollout2_source=legacy forward_noiser_reverse=true forward_noiser_cycle_enabled=false forward_noiser_apply_decoupled=false forward_noiser_apply_gt_former=false forward_noiser_apply_gt_both=false forward_noiser_apply_in_aux=false reverse_noiser_dedrift_enabled=false reverse_noiser_dedrift_apply_to_commit=false reverse_noiser_dedrift_apply_to_flash=false reverse_noiser_dedrift_apply_to_train_score=false reverse_noiser_internalize_weight=0.0 ladd_gt_vs_fake_enabled=false ladd_gt_transition_enabled=true ladd_fake_backbone_grad_scale=0.2 ladd_gt_transition_carn_former=false ladd_gt_transition_carn_latter_reverse=true ladd_gt_transition_carn_steps=1 ladd_gt_transition_gen_detach_former=true"
    ;;
  *)
    echo "[CARN-SURROGATE] CARNMODE must be reference|commit|aux_minus|commit_aux|former_plus|latter_minus; got $CARNMODE" >&2
    exit 5
    ;;
esac

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
COMMON="sample_interval=$SAMPLE_EVERY checkpoint_interval=1000 auto_resume=false save_full_checkpoint=false log_interval=5 memory_audit_enabled=true gan_grad_telemetry_every=5 texture_tripwire_every=5 gan_pixel_texture_enabled=false gan_of_enabled=false fake_alt_head_enabled=false real_teacher_train_online=false dmd_frozen_teacher_pass_enabled=false boundary_vae_roundtrip=false state_probe_aux_enabled=false surrogate_critic_enabled=false surrogate_decoder_shaped_enabled=true surrogate_decoder_shaped_bundle=$BUNDLE surrogate_decoder_shaped_audit_every=0 surrogate_teacher_backbone=ladd_pixel pix_gan_weight=$PIXW pix_r1_gamma=0.0 pix_crop_lat=[24,32] pix_lat_frames_per_crop=3 pix_finish_grad_enabled=false exit_exclude_last_rung=true pix_flash_grad_select_enabled=true ladd_feature_source=vgg ladd_fake_backbone_trainable=false ladd_defer_disc_update=$LADD_DEFER_DISC_UPDATE ladd_disc_loss_weight=0.0 ladd_r1_gamma=$LADD_R1_GAMMA ladd_pixel_crop_rows=24 ladd_pixel_crop_cols=32 ladd_pixel_crops_per_row=2 ladd_pixel_lat_frames=3 ladd_pixel_frames_per_crop=5 ladd_pixel_border_trim=8 ladd_pixel_decode_batch=4 ladd_pixel_decode_split=1 ladd_pixel_decode_cache=false ladd_pixel_crop_stratify_x=1 ladd_pixel_crop_y_lo_frac=0.0 ladd_pixel_crop_y_hi_frac=1.0 ladd_pixel_encoder_trainable=false ladd_pixel_encoder_lr_scale=0.0 ladd_pixel_input_filter=$PIXEL_FILTER ladd_pixel_swt_strength=$SWT_STRENGTH stat_anchor_loss_weight=$STAT_ANCHOR gan_lr=$GANLR gan_updates_per_step=$GANUPDATES ladd_fake_sample_source=flash ladd_disc_force_clean=true flash_dmd_enabled=true flash_dmd_gan_t=60 $CARN_EXTRA"

PIXW_TAG=${PIXW//./p}
SHARE_TAG=${TARGET_SHARE//./p}
GANLR_TAG=${GANLR//./p}
GANLR_TAG=${GANLR_TAG//-/m}
R1_TAG=${LADD_R1_GAMMA//./p}
if [ "$LADD_DEFER_DISC_UPDATE" = "true" ]; then
  DORDER_TAG=ddefer_gthenD
else
  DORDER_TAG=dinline_DthenG
fi
DARM="carnbaseoff_flashgan60_decoderpullback_q1p99_${SURROGATE_STAGE}_${CARNMODE}_share${SHARE_TAG}_${PIXEL_FILTER}_sa${STAT_ANCHOR}_w${PIXW_TAG}_dlr${GANLR_TAG}_du${GANUPDATES}_r1${R1_TAG}_${DORDER_TAG}_${RUNSTAMP}"

echo "[CARN-SURROGATE-CONFIG] source=$SRC base_wandb=g5ndc0fz"
echo "[CARN-SURROGATE-CONFIG] holder=$HOLDER node=$NODE stage=$SURROGATE_STAGE share=$TARGET_SHARE pixw=$PIXW gan_lr=$GANLR gan_updates=$GANUPDATES ladd_r1_gamma=$LADD_R1_GAMMA defer_disc_update=$LADD_DEFER_DISC_UPDATE carn_mode=$CARNMODE carn_aux_weight=$CARN_AUX_WEIGHT pixel_filter=$PIXEL_FILTER swt_strength=$SWT_STRENGTH stat_anchor=$STAT_ANCHOR"
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
  echo "[CARN-SURROGATE-CONFIG] holder=$HOLDER node=$NODE stage=$SURROGATE_STAGE share=$TARGET_SHARE pixw=$PIXW gan_lr=$GANLR gan_updates=$GANUPDATES defer_disc_update=$LADD_DEFER_DISC_UPDATE carn_mode=$CARNMODE carn_aux_weight=$CARN_AUX_WEIGHT pixel_filter=$PIXEL_FILTER swt_strength=$SWT_STRENGTH stat_anchor=$STAT_ANCHOR"
  echo "[CARN-SURROGATE-CONFIG] run_name=dmd10k_${DARM}_j${HOLDER}"
  echo "[CARN-SURROGATE-CONFIG] max_steps=$MAXSTEPS last_wins=[$COMMON]"
} > "$LOG"

set +e
SLURM_JOB_ID=$HOLDER SLURM_NNODES=1 SLURM_JOB_NODELIST=$NODE \
  COMMITDEDRIFT=false DARM=$DARM MAXSTEPS=$MAXSTEPS TELEM=5 \
  bash "$TMP1" >> "$LOG" 2>&1
RC=$?
set -e
echo "[CARN-SURROGATE] exit=$RC log=$LOG"
exit "$RC"
