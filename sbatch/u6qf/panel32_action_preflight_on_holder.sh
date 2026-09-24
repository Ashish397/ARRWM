#!/bin/bash
# Audit four disjoint action-convention shards on a four-GPU holder.
# Usage: panel32_action_preflight_on_holder.sh BATCH_INDEX SHARD_COUNT
set -euo pipefail

BATCH=${1:?preflight batch index required}
SHARDS=${2:?preflight shard count required}
case "$BATCH" in *[!0-9]*|'') echo "invalid batch: $BATCH" >&2; exit 2 ;; esac
case "$SHARDS" in *[!0-9]*|'') echo "invalid shard count: $SHARDS" >&2; exit 2 ;; esac
if [ "$SHARDS" -lt 4 ] || [ $((SHARDS % 4)) -ne 0 ] || \
   [ "$BATCH" -ge $((SHARDS / 4)) ]; then
  echo "invalid preflight batch $BATCH for $SHARDS shards" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/panel32_eval_env.sh"
test -f "$OUT/logs/SETUP_COMPLETE"
mkdir -p "$OUT/preflight" "$OUT/logs/preflight" "$OUT/markers"
rm -f "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE"
require_four_distinct_cuda_ordinals "action_preflight_batch_${BATCH}"
cd "$CODE"
UIDS=$("$PY" - "$CONFIG" <<'PY'
import json, sys
print(",".join(json.load(open(sys.argv[1]))["context_ids"]))
PY
)

pids=()
labels=()
for lane in 0 1 2 3; do
  shard=$((BATCH * 4 + lane))
  complete="$OUT/markers/preflight_shard${shard}.COMPLETE"
  failed="$OUT/markers/preflight_shard${shard}.FAILED"
  log="$OUT/logs/preflight/shard${shard}.log"
  rm -f "$complete" "$failed"
  (CUDA_VISIBLE_DEVICES="$lane" "$PY" grids/eval/preflight_action_conventions.py \
      --out "$OUT/preflight" --uids "$UIDS" --models "$PANEL32_MODELS" \
      --require-positive minwm,minwm_ode,yume5b \
      --shard-index "$shard" --shard-count "$SHARDS" \
      >"$log" 2>&1 \
    && touch "$complete" || { rc=$?; touch "$failed"; exit "$rc"; }) &
  pids+=("$!")
  labels+=("preflight_shard${shard}")
done

rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "FAILED ${labels[$i]}" >&2; rc=1; }
done
if [ "$rc" -ne 0 ]; then exit 1; fi
echo "PANEL32_ACTION_PREFLIGHT_BATCH_COMPLETE batch=$BATCH shards=$SHARDS $(date -Is)"
