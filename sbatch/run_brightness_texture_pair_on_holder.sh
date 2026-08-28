#!/bin/bash
# Run two isolated one-node decoder-pullback arms from inside a durable
# two-node holder command.  The srun clients therefore live with the Slurm
# batch allocation rather than the agent/login session.
set -euo pipefail

: "${ALLOC:?holder loop supplies ALLOC}"
: "${NODE_A:?set first holder node}"
: "${NODE_B:?set second holder node}"
: "${FILTER_A:?none|dc|swt}"
: "${FILTER_B:?none|dc|swt}"
: "${ANCHOR_A:?stat anchor weight for first arm}"
: "${ANCHOR_B:?stat anchor weight for second arm}"
: "${PHASE:?cal or active label}"
: "${PIXW_A:?first arm GAN generator weight}"
: "${PIXW_B:?second arm GAN generator weight}"

RUNSTAMP=${RUNSTAMP:-$(date +%H%M%S)}
MAXSTEPS=${MAXSTEPS:-36}
SAMPLE_EVERY=${SAMPLE_EVERY:-15}
GANLR=${GANLR:-1e-3}
GANUPDATES=${GANUPDATES:-1}
SWT_STRENGTH=${SWT_STRENGTH:-1.0}
SURROGATE_STAGE=${SURROGATE_STAGE:-calibrate}
SURROGATE_Q1_MIN=${SURROGATE_Q1_MIN:-0.992351}
CAL_RUN_A=${CAL_RUN_A:-not_required}
CAL_RUN_B=${CAL_RUN_B:-not_required}
SURROGATE_APPROVED=${SURROGATE_APPROVED:-NO}

HOLDER=$ALLOC NODE=$NODE_A PORTOFF=28101 RUNSTAMP="${RUNSTAMP}_${PHASE}a" \
  MAXSTEPS=$MAXSTEPS SAMPLE_EVERY=$SAMPLE_EVERY GANLR=$GANLR \
  GANUPDATES=$GANUPDATES PIXW=$PIXW_A TARGET_SHARE="${PHASE}_${FILTER_A}" \
  PIXEL_FILTER=$FILTER_A SWT_STRENGTH=$SWT_STRENGTH \
  STAT_ANCHOR=$ANCHOR_A CARNMODE=reference \
  SURROGATE_STAGE=$SURROGATE_STAGE \
  SURROGATE_APPROVED=$SURROGATE_APPROVED \
  SURROGATE_CALIBRATION_RUN=$CAL_RUN_A \
  SURROGATE_Q1_MIN=$SURROGATE_Q1_MIN \
  bash sbatch/run_carn_decoder_surrogate_node.sh &
pid_a=$!

HOLDER=$ALLOC NODE=$NODE_B PORTOFF=28202 RUNSTAMP="${RUNSTAMP}_${PHASE}b" \
  MAXSTEPS=$MAXSTEPS SAMPLE_EVERY=$SAMPLE_EVERY GANLR=$GANLR \
  GANUPDATES=$GANUPDATES PIXW=$PIXW_B TARGET_SHARE="${PHASE}_${FILTER_B}" \
  PIXEL_FILTER=$FILTER_B SWT_STRENGTH=$SWT_STRENGTH \
  STAT_ANCHOR=$ANCHOR_B CARNMODE=reference \
  SURROGATE_STAGE=$SURROGATE_STAGE \
  SURROGATE_APPROVED=$SURROGATE_APPROVED \
  SURROGATE_CALIBRATION_RUN=$CAL_RUN_B \
  SURROGATE_Q1_MIN=$SURROGATE_Q1_MIN \
  bash sbatch/run_carn_decoder_surrogate_node.sh &
pid_b=$!

rc=0
wait "$pid_a" || rc=$?
wait "$pid_b" || rc=$?
exit "$rc"
