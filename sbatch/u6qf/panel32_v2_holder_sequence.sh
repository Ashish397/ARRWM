#!/bin/bash
# Pin holder-dispatched generation/evaluation work to the locked panel32-v2
# sources and staging tree.  This prevents a repair task from falling back to
# the historical v1 defaults in panel32_holder_sequence.sh.
set -euo pipefail

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=${PANEL32_STAGE:-$R/panel32_v2_stage}

export PANEL32_STAGE="$P"
export PANEL32_MANIFEST="${PANEL32_MANIFEST:-$A/grids/eval/panel32_locked_v2.json}"
export PANEL32_PROVENANCE="${PANEL32_PROVENANCE:-$P/sources/panel32_source_provenance.json}"
export PANEL32_BUNDLES="${PANEL32_BUNDLES:-$P/ours_seed_bundles/panel32_seed_bundles.json}"

exec bash "$A/sbatch/u6qf/panel32_holder_sequence.sh" "$@"
