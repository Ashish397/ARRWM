#!/bin/bash
set -u

cd /scratch/u6ex/as1748.u6ex/ARRWM

# Strong-but-stable matched CARN pair: the better-separating R1=1 GAN at the
# same ~1.4x generator weight in both arms. Only aux-minus changes.
HOLDER=6168505 NODE=nid011284 PIXW=.0175 \
TARGET_SHARE=strongstable300_dc_aux_r1g1 CARNMODE=aux_minus \
CARN_AUX_WEIGHT=.25 GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=1 \
LADD_DEFER_DISC_UPDATE=false SURROGATE_FRESH_DISC_ORDER=true \
PIXEL_FILTER=dc STAT_ANCHOR=1.0 SURROGATE_STAGE=active \
SURROGATE_APPROVED=YES SURROGATE_CALIBRATION_RUN=y9ry8jhr \
SURROGATE_Q1_MIN=.992351 MAXSTEPS=300 SAMPLE_EVERY=15 \
RUNSTAMP=2808q1p99freshstrongstable300dcauxr1g1 PORTOFF=29803 \
bash sbatch/run_carn_decoder_surrogate_node.sh \
> logs/carn_surrogate_dispatch_strongstable300_dc_aux_r1g1_h6168505_nid011284.log 2>&1 &
pid0=$!

HOLDER=6168505 NODE=nid011296 PIXW=.0175 \
TARGET_SHARE=strongstable300_dc_ref_r1g1 CARNMODE=reference \
GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=1 \
LADD_DEFER_DISC_UPDATE=false SURROGATE_FRESH_DISC_ORDER=true \
PIXEL_FILTER=dc STAT_ANCHOR=1.0 SURROGATE_STAGE=active \
SURROGATE_APPROVED=YES SURROGATE_CALIBRATION_RUN=y9ry8jhr \
SURROGATE_Q1_MIN=.992351 MAXSTEPS=300 SAMPLE_EVERY=15 \
RUNSTAMP=2808q1p99freshstrongstable300dcrefr1g1 PORTOFF=29804 \
bash sbatch/run_carn_decoder_surrogate_node.sh \
> logs/carn_surrogate_dispatch_strongstable300_dc_ref_r1g1_h6168505_nid011296.log 2>&1 &
pid1=$!

rc=0
wait "$pid0" || rc=$?
wait "$pid1" || rc=$?
exit "$rc"
