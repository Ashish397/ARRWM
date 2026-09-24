#!/bin/bash
# Inject one ODE panel32 arm/action into an existing four-GPU Isambard holder.
set -euo pipefail

ARM=${1:?usage: ode_panel32_on_holder.sh ARM ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX [NODE]}
ACTION=${2:?usage: ode_panel32_on_holder.sh ARM ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX [NODE]}
ALLOC=${3:?usage: ode_panel32_on_holder.sh ARM ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX [NODE]}
PANEL_MANIFEST=${4:?usage: ode_panel32_on_holder.sh ARM ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX [NODE]}
SOURCE_PROVENANCE=${5:?usage: ode_panel32_on_holder.sh ARM ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX [NODE]}
BUNDLE_INDEX=${6:?usage: ode_panel32_on_holder.sh ARM ACTION ALLOC PANEL_MANIFEST SOURCE_PROVENANCE BUNDLE_INDEX [NODE]}
NODE=${7:-$(squeue -h -j "$ALLOC" -o '%N' | head -1)}

case "$ARM" in local_kl|pointwise_mse) ;; *) echo "bad ODE arm: $ARM" >&2; exit 2 ;; esac
case "$ACTION" in F|FR|R|BR|B|BL|L|FL|N) ;; *) echo "bad action: $ACTION" >&2; exit 2 ;; esac
test "$(squeue -h -j "$ALLOC" -o '%T' | head -1)" = RUNNING
test -n "$NODE"

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
exec srun --overlap --external-launcher --jobid="$ALLOC" --nodes=1 \
  --nodelist="$NODE" --ntasks=1 bash \
  "$R/ARRWM/sbatch/u6qf/ode_panel32_holder_action.sh" \
  "$ARM" "$ACTION" "$PANEL_MANIFEST" "$SOURCE_PROVENANCE" "$BUNDLE_INDEX"
