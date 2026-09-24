#!/bin/bash
# Inject one ARRWM panel32 action into a running raw four-GPU holder.
set -euo pipefail
ACTION=${1:?usage: ours_panel32_on_holder.sh ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX [NODE]}
ALLOC=${2:?usage: ours_panel32_on_holder.sh ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX [NODE]}
PANEL_MANIFEST=${3:?usage: ours_panel32_on_holder.sh ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX [NODE]}
SOURCE_PROVENANCE=${4:?usage: ours_panel32_on_holder.sh ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX [NODE]}
BUNDLE_INDEX=${5:?usage: ours_panel32_on_holder.sh ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX [NODE]}
NODE=${6:-$(squeue -h -j "$ALLOC" -o '%N' | head -1)}
test "$(squeue -h -j "$ALLOC" -o '%T' | head -1)" = RUNNING
R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
exec srun --overlap --external-launcher --jobid="$ALLOC" --nodes=1 \
  --nodelist="$NODE" --ntasks=1 bash \
  "$R/ARRWM/sbatch/u6qf/ours_panel32_holder_action.sh" \
  "$ACTION" "$PANEL_MANIFEST" "$SOURCE_PROVENANCE" "$BUNDLE_INDEX"
