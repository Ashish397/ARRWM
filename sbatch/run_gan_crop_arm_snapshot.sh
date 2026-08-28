#!/bin/bash
# Stable entry point for a long holder task.  Bash reads a script lazily, so
# editing the source launcher while it is blocked inside torchrun can make the
# live process resume at a shifted byte offset.  Execute an immutable snapshot
# instead, then remove it after the holder task returns.
set -euo pipefail

ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
SOURCE="$ROOT/sbatch/run_gan_crop_arm_on_holder.sh"
MAXSTEPS=${MAXSTEPS:-200}
SAMPLE_EVERY=${SAMPLE_EVERY:-15}

if ! [[ "$MAXSTEPS" =~ ^[1-9][0-9]*$ && "$SAMPLE_EVERY" =~ ^[1-9][0-9]*$ ]]; then
  echo "[GAN-CROP-SNAPSHOT] FATAL: MAXSTEPS and SAMPLE_EVERY must be positive integers" >&2
  exit 13
fi
if [ "$SAMPLE_EVERY" -ge "$MAXSTEPS" ]; then
  echo "[GAN-CROP-SNAPSHOT] FATAL: SAMPLE_EVERY=$SAMPLE_EVERY must be smaller than MAXSTEPS=$MAXSTEPS; the boundary sample is consumed on the following training step, so this run would produce no pred_image_rollout video" >&2
  exit 14
fi

SNAPSHOT=$(mktemp /tmp/run_gan_crop_arm_on_holder.XXXXXX.sh)
trap 'rm -f "$SNAPSHOT"' EXIT
cp -- "$SOURCE" "$SNAPSHOT"
cd "$ROOT"
bash "$SNAPSHOT" "$@"
