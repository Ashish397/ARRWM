#!/bin/bash
set -u

cd /scratch/u6ex/as1748.u6ex/ARRWM

# Cache-off D->G controls: raw isolates the DC front end, and aux-minus tests
# whether the naturally paired field scale transfers to the CARN winner.
HOLDER=6168505 NODE=nid011284 PIXW=0 \
TARGET_SHARE=cacheoff_inline_cal_raw_r1g10 CARNMODE=reference \
GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=10 \
LADD_DEFER_DISC_UPDATE=false PIXEL_FILTER=none STAT_ANCHOR=1.0 \
MAXSTEPS=45 SAMPLE_EVERY=15 RUNSTAMP=2808cacheoffinlinecalrawr1g10 \
PORTOFF=29603 bash sbatch/run_carn_decoder_surrogate_node.sh \
> logs/carn_surrogate_dispatch_cacheoff_inline_cal_raw_r1g10_h6168505_nid011284.log 2>&1 &
pid0=$!

HOLDER=6168505 NODE=nid011296 PIXW=0 \
TARGET_SHARE=cacheoff_inline_cal_dc_aux_r1g10 CARNMODE=aux_minus \
CARN_AUX_WEIGHT=0.25 GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=10 \
LADD_DEFER_DISC_UPDATE=false PIXEL_FILTER=dc STAT_ANCHOR=1.0 \
MAXSTEPS=45 SAMPLE_EVERY=15 RUNSTAMP=2808cacheoffinlinecaldcauxr1g10 \
PORTOFF=29604 bash sbatch/run_carn_decoder_surrogate_node.sh \
> logs/carn_surrogate_dispatch_cacheoff_inline_cal_dc_aux_r1g10_h6168505_nid011296.log 2>&1 &
pid1=$!

rc=0
wait "$pid0" || rc=$?
wait "$pid1" || rc=$?
exit "$rc"
