#!/bin/bash
# Generate the four corrected v3 contexts for one ARRWM arm/action.
set -euo pipefail

KIND=${1:?usage: panel32_v3_arrwm_delta_task.sh KIND ARM ACTION}
ARM=${2:?usage: panel32_v3_arrwm_delta_task.sh KIND ARM ACTION}
ACTION=${3:?usage: panel32_v3_arrwm_delta_task.sh KIND ARM ACTION}
case "$KIND" in ours|ode) ;; *) echo "bad kind: $KIND" >&2; exit 2 ;; esac
case "$ACTION" in F|FR|R|BR|B|BL|L|FL|N) ;; *) echo "bad action: $ACTION" >&2; exit 2 ;; esac

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=${PANEL32_STAGE:-$R/panel32_v3_stage}
MANIFEST=$A/grids/eval/panel32_locked_v3.json
PROVENANCE=$P/sources/panel32_source_provenance.json
BUNDLES=$P/ours_seed_bundles/panel32_seed_bundles.json

test -f "$P/logs/seed_setup/COMPLETE"
test -s "$BUNDLES"
export PANEL32_STAGE="$P"
export PANEL32_MANIFEST="$MANIFEST"
export PANEL32_PROVENANCE="$PROVENANCE"
export PANEL32_BUNDLES="$BUNDLES"
export PANEL32_CONTEXT_SHARDS="8 12 13 15"
export PANEL32_CONTEXT_SHARD_COUNT=32

bash "$A/sbatch/u6qf/panel32_holder_sequence.sh" "$KIND:$ARM:$ACTION"

