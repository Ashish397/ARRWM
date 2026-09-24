#!/bin/bash
# Score four disjoint conjuration-review shards, one process per H100.
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
require_four_distinct_cuda_ordinals "conjuration_v2_vlm_batch_${BATCH}"
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
CANDIDATES=${PANEL32_CONJURATION_V2_CANDIDATES:-$OUT/conjuration_v2_audit/conjuration_v2_adjudication.csv}
DEST=${PANEL32_CONJURATION_V2_TRIAGE_DEST:-$OUT/conjuration_v2_audit/vlm_triage}
mkdir -p "$DEST/logs" "$DEST/markers"
test -s "$CANDIDATES"
cd "$CODE"

pids=()
labels=()
for lane in 0 1 2 3; do
  shard=$((BATCH * 4 + lane))
  complete="$DEST/markers/shard${shard}.COMPLETE"
  failed="$DEST/markers/shard${shard}.FAILED"
  log="$DEST/logs/shard${shard}.log"
  output="$DEST/shard${shard}.csv"
  rm -f "$complete" "$failed"
  (CUDA_VISIBLE_DEVICES="$lane" "$PY" \
      grids/eval/panel32_conjuration_v2_vlm_triage.py \
      --candidates "$CANDIDATES" --output "$output" \
      --shard-index "$shard" --shard-count "$SHARDS" >"$log" 2>&1 \
    && touch "$complete" || { rc=$?; touch "$failed"; exit "$rc"; }) &
  pids+=("$!")
  labels+=("shard${shard}")
done

rc=0
for index in "${!pids[@]}"; do
  wait "${pids[$index]}" || { echo "FAILED ${labels[$index]}" >&2; rc=1; }
done
test "$rc" -eq 0
echo "PANEL32_CONJURATION_V2_VLM_BATCH_COMPLETE batch=$BATCH shards=$SHARDS $(date -Is)"
