#!/bin/bash
# Weight-zero production-geometry calibration for the promoted staged CARN
# candidate. This wrapper does not submit or cancel Slurm jobs; run it through
# a durable holder command with HOLDER and NODE set to one allocated node.
set -euo pipefail

: "${HOLDER:?set HOLDER to the running durable holder job id}"
: "${NODE:?set NODE to one node allocated to HOLDER}"

HOLDER="$HOLDER" NODE="$NODE" \
PIXW=0 TARGET_SHARE=commitauxstaged_w0 \
CARNMODE=commit_aux_staged CARN_AUX_WEIGHT=.25 \
CARN_COMMIT_ALPHA=.25 CARN_COMMIT_START_STEP=100 \
CARN_COMMIT_RAMP_STEPS=100 \
PIXEL_FILTER=dc SWT_STRENGTH=0.0 STAT_ANCHOR=1.0 \
GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=1 \
LADD_DEFER_DISC_UPDATE=false SURROGATE_FRESH_DISC_ORDER=true \
SURROGATE_STAGE=calibrate MAXSTEPS="${MAXSTEPS:-225}" \
SAMPLE_EVERY="${SAMPLE_EVERY:-15}" \
RUNSTAMP="${RUNSTAMP:-2808q1p99commitauxstagedw0}" \
PORTOFF="${PORTOFF:-30117}" \
bash sbatch/run_carn_decoder_surrogate_node.sh
