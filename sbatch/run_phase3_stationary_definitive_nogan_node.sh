#!/bin/bash
# Matched no-GAN ablation of the definitive stationary Phase 3 recipe.
# It keeps the phase-2 step-400 source, seven-chunk geometry, CARN F/G cycle,
# aux-minus/internalisation, staged commit, stat anchor, Flash DMD, and saves
# full checkpoints at steps 100 and 200.  The adversarial critic and its
# decoder-shaped pullback are disabled by last-wins overrides.
set -euo pipefail

: "${HOLDER:?set HOLDER to the live two-node holder id}"
: "${NODE:?set NODE to one node allocated to HOLDER}"

HOLDER="$HOLDER" NODE="$NODE" \
GAN_ENABLED=false \
PIXW=.0175 TARGET_SHARE=definitive_htfhu37j_harmony_dcoff_nogan \
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
RUNSTAMP="${RUNSTAMP:-2808phase3stationarydefinitive_nogan}" \
PORTOFF="${PORTOFF:-31682}" \
bash sbatch/run_carn_decoder_surrogate_node.sh
