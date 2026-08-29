#!/bin/bash
set -u

cd /scratch/u6ex/as1748.u6ex/ARRWM

# Matched 3%-median controls: raw isolates the DC front end; aux-minus changes
# only the recent CARN internalization consumer relative to the DC reference.
HOLDER=6168505 NODE=nid011284 PIXW=.032 \
TARGET_SHARE=active3pct_raw_ref_r1g10 CARNMODE=reference \
GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=10 \
LADD_DEFER_DISC_UPDATE=false SURROGATE_FRESH_DISC_ORDER=true \
PIXEL_FILTER=none STAT_ANCHOR=1.0 SURROGATE_STAGE=active \
SURROGATE_APPROVED=YES SURROGATE_CALIBRATION_RUN=bbj4flwh \
SURROGATE_Q1_MIN=.992351 MAXSTEPS=100 SAMPLE_EVERY=15 \
RUNSTAMP=2808q1p99freshactive3pctrawrefr1g10 PORTOFF=29703 \
bash sbatch/run_carn_decoder_surrogate_node.sh \
> logs/carn_surrogate_dispatch_active3pct_raw_ref_r1g10_h6168505_nid011284.log 2>&1 &
pid0=$!

HOLDER=6168505 NODE=nid011296 PIXW=.040 \
TARGET_SHARE=active3pct_dc_aux_r1g10 CARNMODE=aux_minus \
CARN_AUX_WEIGHT=.25 GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=10 \
LADD_DEFER_DISC_UPDATE=false SURROGATE_FRESH_DISC_ORDER=true \
PIXEL_FILTER=dc STAT_ANCHOR=1.0 SURROGATE_STAGE=active \
SURROGATE_APPROVED=YES SURROGATE_CALIBRATION_RUN=bfvw2wik \
SURROGATE_Q1_MIN=.992351 MAXSTEPS=100 SAMPLE_EVERY=15 \
RUNSTAMP=2808q1p99freshactive3pctdcauxr1g10 PORTOFF=29704 \
bash sbatch/run_carn_decoder_surrogate_node.sh \
> logs/carn_surrogate_dispatch_active3pct_dc_aux_r1g10_h6168505_nid011296.log 2>&1 &
pid1=$!

rc=0
wait "$pid0" || rc=$?
wait "$pid1" || rc=$?
exit "$rc"
