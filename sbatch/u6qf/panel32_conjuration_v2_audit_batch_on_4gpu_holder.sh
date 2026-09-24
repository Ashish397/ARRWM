#!/bin/bash
# Run four disjoint fail-closed conjuration-v2 audit shards on an existing
# four-GPU panel32 holder. This writes a new audit tree; published outputs are
# never touched.
set -euo pipefail

BATCH=${1:?batch index required}
SHARDS=${2:-48}
case "$BATCH" in *[!0-9]*|'') echo "invalid batch: $BATCH" >&2; exit 2 ;; esac
case "$SHARDS" in *[!0-9]*|'') echo "invalid shard count: $SHARDS" >&2; exit 2 ;; esac
if [ "$SHARDS" -lt 4 ] || [ $((SHARDS % 4)) -ne 0 ] || [ "$BATCH" -ge $((SHARDS / 4)) ]; then
  echo "invalid batch $BATCH for $SHARDS shards" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/panel32_eval_env.sh"
test -s "$OUT/video_manifest.csv"
require_four_distinct_cuda_ordinals "conjuration_v2_audit_batch_${BATCH}"

AUDIT_OUT=${PANEL32_CONJURATION_V2_OUT:-$OUT/conjuration_v2_audit}
mkdir -p "$AUDIT_OUT/logs" "$AUDIT_OUT/markers"
cd "$CODE"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8

pids=()
labels=()
for lane in 0 1 2 3; do
  shard=$((BATCH * 4 + lane))
  complete="$AUDIT_OUT/markers/shard${shard}.COMPLETE"
  failed="$AUDIT_OUT/markers/shard${shard}.FAILED"
  log="$AUDIT_OUT/logs/shard${shard}.log"
  rm -f "$complete" "$failed"
  (CUDA_VISIBLE_DEVICES="$lane" "$PY" \
      grids/eval/panel32_conjuration_v2_audit.py \
      --eval-root "$OUT" --output "$AUDIT_OUT" \
      --windows 0,6,12,18,24 \
      --shard-index "$shard" --shard-count "$SHARDS" \
      >"$log" 2>&1 \
    && touch "$complete" || { rc=$?; touch "$failed"; exit "$rc"; }) &
  pids+=("$!")
  labels+=("conjuration_v2_shard${shard}")
done

rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "FAILED ${labels[$i]}" >&2; rc=1; }
done
if [ "$rc" -ne 0 ]; then exit 1; fi
echo "PANEL32_CONJURATION_V2_BATCH_COMPLETE batch=$BATCH shards=$SHARDS $(date -Is)"
