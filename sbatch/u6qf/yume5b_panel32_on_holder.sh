#!/bin/bash
# Login-side injector for one action into a raw running four-GPU holder.
# Usage: bash yume5b_panel32_on_holder.sh ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE [NODE]
# This script never submits or cancels an allocation.
set -euo pipefail

ACTION=${1:?usage: yume5b_panel32_on_holder.sh ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE [NODE]}
ALLOC=${2:?usage: yume5b_panel32_on_holder.sh ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE [NODE]}
PANEL_MANIFEST=${3:?usage: yume5b_panel32_on_holder.sh ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE [NODE]}
SOURCE_PROVENANCE=${4:?usage: yume5b_panel32_on_holder.sh ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE [NODE]}
NODE=${5:-}
case "$ACTION" in F|FR|R|BR|B|BL|L|FL|N) ;; *) echo "bad action: $ACTION" >&2; exit 2 ;; esac
case "$ALLOC" in *[!0-9]*|'') echo "ALLOC must be a raw numeric allocation ID: $ALLOC" >&2; exit 2 ;; esac
test -f "$PANEL_MANIFEST"
test -f "$SOURCE_PROVENANCE"

STATE=$(squeue -h -j "$ALLOC" -o '%T' | head -1)
if [ "$STATE" != "RUNNING" ]; then
  echo "allocation $ALLOC is not RUNNING (state=${STATE:-not-in-squeue})" >&2
  exit 1
fi
if [ -z "$NODE" ]; then
  NODE=$(squeue -h -j "$ALLOC" -o '%N' | head -1)
fi
if [ -z "$NODE" ] || [ "$NODE" = "(null)" ]; then
  echo "could not resolve node for allocation $ALLOC; pass NODE explicitly" >&2
  exit 1
fi

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
PAYLOAD=${YUME_PANEL_HOLDER_PAYLOAD:-$R/ARRWM/sbatch/u6qf/yume5b_panel32_holder_action.sh}
test -f "$PAYLOAD"
echo "YUME_PANEL32_HOLDER_INJECT action=$ACTION alloc=$ALLOC node=$NODE mode=external-launcher $(date -Is)"

exec srun \
  --overlap \
  --external-launcher \
  --jobid="$ALLOC" \
  --nodes=1 \
  --nodelist="$NODE" \
  --ntasks=1 \
  bash "$PAYLOAD" "$ACTION" "$PANEL_MANIFEST" "$SOURCE_PROVENANCE"
