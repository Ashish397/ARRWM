#!/bin/bash
# Run four disjoint shards of one metric inside an existing four-GPU holder.
# Usage: panel32_eval_metric_batch_on_4gpu_holder.sh TASK BATCH_INDEX SHARD_COUNT
# TASK: cpu, style, control, geometry, conjuration, conjuration-v2, or longreloc.
set -euo pipefail

TASK=${1:?metric task required}
BATCH=${2:?batch index required}
SHARDS=${3:?shard count required}
case "$TASK" in cpu|style|control|geometry|conjuration|conjuration-v2|longreloc) ;; *) echo "invalid task: $TASK" >&2; exit 2 ;; esac
case "$BATCH" in *[!0-9]*|'') echo "invalid batch: $BATCH" >&2; exit 2 ;; esac
case "$SHARDS" in *[!0-9]*|'') echo "invalid shard count: $SHARDS" >&2; exit 2 ;; esac
if [ "$SHARDS" -lt 4 ] || [ $((SHARDS % 4)) -ne 0 ] || [ "$BATCH" -ge $((SHARDS / 4)) ]; then
  echo "invalid batch $BATCH for $SHARDS shards" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/panel32_eval_env.sh"
test -f "$OUT/logs/SETUP_COMPLETE"
test -f "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE"
"$PY" - "$OUT/logs/shard_plan.json" "$SHARDS" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
assert plan["shard_count"] == int(sys.argv[2])
assert plan["shards_per_holder"] == 4
PY
if [ "$TASK" != cpu ] && [ "$TASK" != longreloc ]; then
  require_four_distinct_cuda_ordinals "${TASK}_batch_${BATCH}"
fi

if [ "$TASK" = longreloc ]; then
  # Four lanes each start four OpenCV workers.  Avoid nested native thread
  # pools oversubscribing the 64-CPU holder.
  export OMP_NUM_THREADS=1
  export OPENBLAS_NUM_THREADS=1
  export MKL_NUM_THREADS=1
else
  export OMP_NUM_THREADS=8
  export OPENBLAS_NUM_THREADS=8
  export MKL_NUM_THREADS=8
fi
cd "$CODE"
mkdir -p "$OUT/logs/$TASK" "$OUT/markers"

pids=()
labels=()
for lane in 0 1 2 3; do
  shard=$((BATCH * 4 + lane))
  complete="$OUT/markers/${TASK}_shard${shard}.COMPLETE"
  failed="$OUT/markers/${TASK}_shard${shard}.FAILED"
  log="$OUT/logs/$TASK/shard${shard}.log"
  rm -f "$complete" "$failed"
  case "$TASK" in
    cpu)
      ("$PY" grids/eval/final_v2_cpu.py --out "$OUT" \
          --shard-index "$shard" --shard-count "$SHARDS" >"$log" 2>&1 \
        && touch "$complete" || { rc=$?; touch "$failed"; exit "$rc"; }) &
      ;;
    style)
      (CUDA_VISIBLE_DEVICES="$lane" "$PY" grids/eval/final_v2_style.py --out "$OUT" \
          --shard-index "$shard" --shard-count "$SHARDS" >"$log" 2>&1 \
        && touch "$complete" || { rc=$?; touch "$failed"; exit "$rc"; }) &
      ;;
    control)
      (CUDA_VISIBLE_DEVICES="$lane" "$PY" grids/eval/final_v2_control.py --out "$OUT" \
          --shard-index "$shard" --shard-count "$SHARDS" >"$log" 2>&1 \
        && touch "$complete" || { rc=$?; touch "$failed"; exit "$rc"; }) &
      ;;
    geometry|conjuration)
      (CUDA_VISIBLE_DEVICES="$lane" "$PY" grids/eval/final_v3_quality_gpu.py --out "$OUT" \
          --metric "$TASK" --shard-index "$shard" --shard-count "$SHARDS" >"$log" 2>&1 \
        && touch "$complete" || { rc=$?; touch "$failed"; exit "$rc"; }) &
      ;;
    conjuration-v2)
      (CUDA_VISIBLE_DEVICES="$lane" "$PY" grids/eval/final_v3_quality_gpu.py --out "$OUT" \
          --metric conjuration --conjuration-profile persistent-salient-v2 \
          --destination-name conjuration_v2_candidates \
          --shard-index "$shard" --shard-count "$SHARDS" >"$log" 2>&1 \
        && touch "$complete" || { rc=$?; touch "$failed"; exit "$rc"; }) &
      ;;
    longreloc)
      ("$PY" grids/eval/iclr_long_relocation.py \
          --manifest "$OUT/video_manifest.csv" --out "$OUT/long_relocation" \
          --shard-index "$shard" --shard-count "$SHARDS" --workers 4 \
          >"$log" 2>&1 \
        && touch "$complete" || { rc=$?; touch "$failed"; exit "$rc"; }) &
      ;;
  esac
  pids+=("$!")
  labels+=("${TASK}_shard${shard}")
done

rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "FAILED ${labels[$i]}" >&2; rc=1; }
done
if [ "$rc" -ne 0 ]; then exit 1; fi
echo "PANEL32_METRIC_BATCH_COMPLETE task=$TASK batch=$BATCH shards=$SHARDS $(date -Is)"
