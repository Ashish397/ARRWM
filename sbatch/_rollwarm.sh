#!/bin/bash
# Warm-start a ROLLING arm from a stationary DMD3 phase-1 checkpoint.
# _maybe_resume globs "phase1_step*.pt" in the run's OWN log_dir, so we seed
# that directory with the source checkpoint and flip auto_resume back on
# (the holder launchers force it off to prevent accidental continuation).
set -e
cd /scratch/u6ex/as1748.u6ex/ARRWM
HOLDER=${HOLDER:?}; PORTOFF=${PORTOFF:?}; DARM=${DARM:?}; SRC=${SRC:?}
RUNSTAMP=${RUNSTAMP:-$(date +%H%M%S)}
DEST="logs/dmd10k_${DARM}/dmd10k_${DARM}_h${HOLDER}_${RUNSTAMP}"
mkdir -p "$DEST"
cp "$SRC" "$DEST/phase1_step0000200.pt"
echo "seeded $DEST with $(basename $SRC) ($(du -h $SRC | cut -f1))"
export HOLDER PORTOFF RUNSTAMP DARM
# MAXSTEPS must EXCEED the resumed step (200) or the loop exits at once --
# that is exactly what silently no-op'd the first attempt. 300 = 100 more.
export STAT_ANCHOR=1.0 MAXSTEPS=${MAXSTEPS:-300} CKPT_EVERY=100000 WAVELET=true
export ODE_CKPT=${ODE_CKPT:?}
export DEXTRA="auto_resume=true ${EXTRA:-}"
exec bash sbatch/_roll_holder.sh
