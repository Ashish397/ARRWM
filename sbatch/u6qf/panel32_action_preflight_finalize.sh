#!/bin/bash
# Merge and fail-closed validate all action-convention preflight shards.
set -euo pipefail

SHARDS=${1:-${PANEL32_EVAL_SHARDS:-48}}
case "$SHARDS" in *[!0-9]*|'') echo "invalid shard count: $SHARDS" >&2; exit 2 ;; esac
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/panel32_eval_env.sh"
test -f "$OUT/logs/SETUP_COMPLETE"
for ((shard=0; shard<SHARDS; shard++)); do
  test -f "$OUT/markers/preflight_shard${shard}.COMPLETE"
  test ! -f "$OUT/markers/preflight_shard${shard}.FAILED"
done
cd "$CODE"
UIDS=$("$PY" - "$CONFIG" <<'PY'
import json, sys
print(",".join(json.load(open(sys.argv[1]))["context_ids"]))
PY
)
rm -f "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE"
"$PY" grids/eval/preflight_action_conventions.py \
  --out "$OUT/preflight" --uids "$UIDS" --models "$PANEL32_MODELS" \
  --require-positive minwm,minwm_ode,yume5b \
  --shard-index 0 --shard-count "$SHARDS" --finalize-only \
  >"$OUT/logs/action_preflight_finalize.log" 2>&1
test -f "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE"
echo "PANEL32_ACTION_PREFLIGHT_COMPLETE shards=$SHARDS $(date -Is)"
