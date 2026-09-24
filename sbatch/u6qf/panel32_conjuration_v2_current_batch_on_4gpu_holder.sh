#!/bin/bash
# Run one four-shard batch of the corrected conjuration audit on only the 14
# source contexts that remain accepted while replacement sources are reviewed.
# SpatialVID is excluded because its symbol-bearing inputs are being replaced.
set -euo pipefail

BATCH=${1:?batch index required}
SHARDS=${2:-48}
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/panel32_eval_env.sh"

export PANEL32_CONJURATION_V2_KEPT_ROOT="$OUT/conjuration_v2_current_input"
export PANEL32_CONJURATION_V2_KEPT_OUT="$OUT/conjuration_v2_audit_current"

exec "$SCRIPT_DIR/panel32_conjuration_v2_kept_batch_on_4gpu_holder.sh" \
  "$BATCH" "$SHARDS"
