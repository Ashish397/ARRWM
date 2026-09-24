#!/bin/bash
# Merge the 30 direct-relocation shards, finalize reviewed conjuration, and
# resume the fail-closed evaluation finisher on one retained holder.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/panel32_eval_env.sh"
SHARDS=${PANEL32_DIRECT_RELOCATION_SHARDS:-30}
case "$SHARDS" in *[!0-9]*|'') echo "invalid shard count: $SHARDS" >&2; exit 2 ;; esac
if [ "$SHARDS" -lt 1 ]; then
  echo "shard count must be positive" >&2
  exit 2
fi
AUDIT="$OUT/conjuration_v2_audit"
FINAL="$OUT/conjuration_v2_final"

cd "$CODE"
"$PY" grids/eval/panel32_direct_relocation_shard.py \
  --out "$OUT" --shard-count "$SHARDS" --merge
"$PY" grids/eval/final_v2_relocation.py --out "$OUT"
"$PY" grids/eval/panel32_conjuration_v2_review.py finalize \
  --manifest "$OUT/video_manifest.csv" \
  --adjudication "$AUDIT/conjuration_v2_adjudication.csv" \
  --output "$FINAL"
bash "$SCRIPT_DIR/panel32_eval_finish.sh"
