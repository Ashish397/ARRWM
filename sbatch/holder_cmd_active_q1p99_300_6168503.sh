#!/bin/bash
set -u

cd /scratch/u6ex/as1748.u6ex/ARRWM

# 300-step evolution wave, promoted after every post-initial 100-step-smoke
# probe through step 66 showed correct D ordering and intact safety telemetry.
HOLDER=6168503 NODE=nid011253 PIXW=.035 \
TARGET_SHARE=active3pct300_dc_ref_r1g10 CARNMODE=reference \
GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=10 \
LADD_DEFER_DISC_UPDATE=false SURROGATE_FRESH_DISC_ORDER=true \
PIXEL_FILTER=dc STAT_ANCHOR=1.0 SURROGATE_STAGE=active \
SURROGATE_APPROVED=YES SURROGATE_CALIBRATION_RUN=83wlndxr \
SURROGATE_Q1_MIN=.992351 MAXSTEPS=300 SAMPLE_EVERY=15 \
RUNSTAMP=2808q1p99freshactive3pct300dcrefr1g10 PORTOFF=29801 \
bash sbatch/run_carn_decoder_surrogate_node.sh \
> logs/carn_surrogate_dispatch_active3pct300_dc_ref_r1g10_h6168503_nid011253.log 2>&1 &
pid0=$!

HOLDER=6168503 NODE=nid011263 PIXW=.0125 \
TARGET_SHARE=active3pct300_dc_ref_r1g1 CARNMODE=reference \
GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=1 \
LADD_DEFER_DISC_UPDATE=false SURROGATE_FRESH_DISC_ORDER=true \
PIXEL_FILTER=dc STAT_ANCHOR=1.0 SURROGATE_STAGE=active \
SURROGATE_APPROVED=YES SURROGATE_CALIBRATION_RUN=y9ry8jhr \
SURROGATE_Q1_MIN=.992351 MAXSTEPS=300 SAMPLE_EVERY=15 \
RUNSTAMP=2808q1p99freshactive3pct300dcrefr1g1 PORTOFF=29802 \
bash sbatch/run_carn_decoder_surrogate_node.sh \
> logs/carn_surrogate_dispatch_active3pct300_dc_ref_r1g1_h6168503_nid011263.log 2>&1 &
pid1=$!

rc=0
wait "$pid0" || rc=$?
wait "$pid1" || rc=$?
exit "$rc"
