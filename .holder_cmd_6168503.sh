#!/bin/bash
set -u

cd /scratch/u6ex/as1748.u6ex/ARRWM

# Cache-off, fresh-D-on-current-batch calibration pair. These are matched to
# the completed deferred controls except for D->G ordering.
HOLDER=6168503 NODE=nid011253 PIXW=0 \
TARGET_SHARE=cacheoff_inline_cal_dc_r1g10 CARNMODE=reference \
GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=10 \
LADD_DEFER_DISC_UPDATE=false PIXEL_FILTER=dc STAT_ANCHOR=1.0 \
MAXSTEPS=45 SAMPLE_EVERY=15 RUNSTAMP=2808cacheoffinlinecaldcr1g10 \
PORTOFF=29601 bash sbatch/run_carn_decoder_surrogate_node.sh \
> logs/carn_surrogate_dispatch_cacheoff_inline_cal_dc_r1g10_h6168503_nid011253.log 2>&1 &
pid0=$!

HOLDER=6168503 NODE=nid011263 PIXW=0 \
TARGET_SHARE=cacheoff_inline_cal_dc_r1g1 CARNMODE=reference \
GANLR=1e-3 GANUPDATES=1 LADD_R1_GAMMA=1 \
LADD_DEFER_DISC_UPDATE=false PIXEL_FILTER=dc STAT_ANCHOR=1.0 \
MAXSTEPS=45 SAMPLE_EVERY=15 RUNSTAMP=2808cacheoffinlinecaldcr1g1 \
PORTOFF=29602 bash sbatch/run_carn_decoder_surrogate_node.sh \
> logs/carn_surrogate_dispatch_cacheoff_inline_cal_dc_r1g1_h6168503_nid011263.log 2>&1 &
pid1=$!

rc=0
wait "$pid0" || rc=$?
wait "$pid1" || rc=$?
exit "$rc"
