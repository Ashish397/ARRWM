#!/bin/bash
# Stable entry point for a long CARN-VX holder task. Bash reads scripts lazily;
# execute an immutable copy so later launcher edits cannot alter a live run.
set -euo pipefail

ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
SOURCE="$ROOT/sbatch/run_carn_vx_on_holder.sh"
SNAPSHOT=$(mktemp /tmp/run_carn_vx_on_holder.XXXXXX.sh)
trap 'rm -f "$SNAPSHOT"' EXIT
cp -- "$SOURCE" "$SNAPSHOT"
cd "$ROOT"
bash "$SNAPSHOT" "$@"
