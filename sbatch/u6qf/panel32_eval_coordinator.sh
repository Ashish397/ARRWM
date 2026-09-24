#!/bin/bash
# Advance the balanced evaluation through its gated parallel phases.
# Setup must already have been queued. This script never retries a failed
# shard; it stops so the failure can be inspected rather than overwritten.
set -euo pipefail

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
CODE=${ARRWM_CODE_ROOT:-$R/ARRWM}
OUT=${PANEL32_EVAL_OUT:-$R/panel32_stage/eval_final}
LOGS="$R/panel32_stage/logs/holders"
PY="$R/miniforge3/envs/arrwm/bin/python"
DISPATCH="$CODE/grids/eval/panel32_dispatch_eval.py"
SHARDS=${PANEL32_EVAL_SHARDS:-48}
POLL=${PANEL32_COORDINATOR_POLL_S:-30}

dispatch() {
  "$PY" "$DISPATCH" "$1" --root "$R" --code "$CODE" --shards "$SHARDS"
}

failed_holder_phase() {
  local pattern=${1:?phase pattern required}
  local file
  for file in "$LOGS"/.holder_done_*_rc*.sh; do
    [ -e "$file" ] || continue
    case "$file" in *_rc0.sh) continue ;; esac
    if grep -Eq "$pattern" "$file"; then
      echo "FAILED_HOLDER_COMMAND file=$file phase_pattern=$pattern" >&2
      return 0
    fi
  done
  return 1
}

any_failed_marker() {
  local pattern=${1:?marker glob required}
  compgen -G "$OUT/markers/$pattern" >/dev/null
}

echo "PANEL32_COORDINATOR_START $(date -Is)"
while [ ! -f "$OUT/logs/SETUP_COMPLETE" ]; do
  if failed_holder_phase 'eval:setup'; then exit 1; fi
  sleep "$POLL"
done
echo "PANEL32_COORDINATOR_SETUP_COMPLETE $(date -Is)"

while [ ! -f "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE" ]; do
  if any_failed_marker 'preflight_shard*.FAILED' || \
     failed_holder_phase 'eval:preflight(:|-finalize)'; then
    exit 1
  fi
  complete=$(find "$OUT/markers" -maxdepth 1 -name 'preflight_shard*.COMPLETE' | wc -l)
  if [ "$complete" -eq "$SHARDS" ]; then
    dispatch preflight-finalize
  else
    dispatch preflight
  fi
  sleep "$POLL"
done
echo "PANEL32_COORDINATOR_PREFLIGHT_COMPLETE $(date -Is)"

while [ ! -f "$OUT/logs/EVALUATION_COMPLETE" ]; do
  if any_failed_marker '*.FAILED' || failed_holder_phase 'eval:(cpu|style|control|geometry|conjuration|longreloc|finish)'; then
    exit 1
  fi
  complete=$(find "$OUT/markers" -maxdepth 1 -type f \
    \( -name 'cpu_shard*.COMPLETE' -o -name 'style_shard*.COMPLETE' \
       -o -name 'control_shard*.COMPLETE' -o -name 'geometry_shard*.COMPLETE' \
       -o -name 'conjuration_shard*.COMPLETE' -o -name 'longreloc_shard*.COMPLETE' \) \
    | wc -l)
  expected=$((6 * SHARDS))
  if [ "$complete" -eq "$expected" ]; then
    dispatch finish
  else
    dispatch metrics
  fi
  sleep "$POLL"
done
echo "PANEL32_COORDINATOR_EVALUATION_COMPLETE $(date -Is)"
