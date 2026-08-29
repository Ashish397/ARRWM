#!/bin/bash
# Definitive Phase 3 stationary run, promoted 2026-08-28.
#
# Phase 2 source: KL ODE step 400.
# Base treatment: htfhu37j.
# Additions: harmony bridge + DC-off/raw-forward high-pass brightness policy.
# Geometry: the first seven chunks only (one seed + six AR chunks), with no
# rolling-window phase. Full resumable checkpoints are written at 100 and 200.
set -euo pipefail

: "${HOLDER:?set HOLDER to the live two-node holder id}"
: "${NODE:?set NODE to one node allocated to HOLDER}"

HOLDER="$HOLDER" NODE="$NODE" \
PIXW=.0175 TARGET_SHARE=definitive_htfhu37j_harmony_dcoff \
CARNMODE=harmony_bridge CARN_AUX_WEIGHT=.25 \
CARN_COMMIT_ALPHA=.25 CARN_COMMIT_START_STEP=100 \
CARN_COMMIT_RAMP_STEPS=100 \
PIXEL_FILTER=raw_grad_hp SWT_STRENGTH=0.0 STAT_ANCHOR=1.0 \
GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=1 \
LADD_DEFER_DISC_UPDATE=false SURROGATE_FRESH_DISC_ORDER=true \
SURROGATE_STAGE=active SURROGATE_APPROVED=YES \
SURROGATE_CALIBRATION_RUN=exact_residual_ladder_2808 \
SURROGATE_Q1_MIN=.992351 \
CARN_STAGED_CALIBRATION_APPROVED=YES \
CARN_STAGED_CALIBRATION_RUN=udw86qaq \
ACTION_CRITIC_FREEZE=true ACTION_TEACHER_MODE=off \
ACTION_CRITIC_GUIDANCE_WEIGHT=.3 \
STATIONARY_FIRST7=true MAXSTEPS=200 SAMPLE_EVERY=15 \
CHECKPOINT_EVERY=100 SAVE_FULL_CHECKPOINT=true \
KEEP_LAST_N_CHECKPOINTS=4 \
RUNSTAMP="${RUNSTAMP:-2808phase3stationarydefinitive}" \
PORTOFF="${PORTOFF:-31517}" \
bash sbatch/run_carn_decoder_surrogate_node.sh
