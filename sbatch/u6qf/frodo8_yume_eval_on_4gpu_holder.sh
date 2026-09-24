#!/bin/bash
# Run one disjoint metric shard inside an already allocated four-GPU holder.
# Usage: frodo8_yume_eval_on_4gpu_holder.sh SHARD_INDEX SHARD_COUNT
set -euo pipefail

SHARD=${1:?shard index required}
SHARDS=${2:?shard count required}
case "$SHARD" in *[!0-9]*|'') exit 2 ;; esac
case "$SHARDS" in *[!0-9]*|'') exit 2 ;; esac
if [ "$SHARDS" -lt 1 ] || [ "$SHARD" -ge "$SHARDS" ]; then
  echo "invalid shard $SHARD/$SHARDS" >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/frodo8_yume_eval_env.sh"
test -f "$OUT/logs/SETUP_COMPLETE"
test -f "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE"
"$PY" - "$OUT/logs/shard_plan.json" "$SHARDS" <<'PY'
import json, sys
assert json.load(open(sys.argv[1]))["shard_count"] == int(sys.argv[2])
PY
require_four_distinct_cuda_ordinals "metric_shard_${SHARD}"

LOG="$OUT/logs/shard${SHARD}"
mkdir -p "$LOG"
rm -f "$LOG/COMPLETE" "$LOG/FAILED"
export HF_HOME="$R/iclrv3_hf_cache"
export TORCH_HOME="$R/torch_home"
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
cd "$CODE"

pids=()
names=()
"$PY" grids/eval/final_v2_cpu.py --out "$OUT" \
  --shard-index "$SHARD" --shard-count "$SHARDS" >"$LOG/cpu.log" 2>&1 &
pids+=("$!"); names+=(cpu_hf)
CUDA_VISIBLE_DEVICES=0 "$PY" grids/eval/final_v2_style.py --out "$OUT" \
  --shard-index "$SHARD" --shard-count "$SHARDS" >"$LOG/style.log" 2>&1 &
pids+=("$!"); names+=(style)
CUDA_VISIBLE_DEVICES=1 "$PY" grids/eval/final_v2_control.py --out "$OUT" \
  --shard-index "$SHARD" --shard-count "$SHARDS" >"$LOG/control.log" 2>&1 &
pids+=("$!"); names+=(control)
CUDA_VISIBLE_DEVICES=2 "$PY" grids/eval/final_v3_quality_gpu.py --out "$OUT" \
  --metric geometry --shard-index "$SHARD" --shard-count "$SHARDS" \
  >"$LOG/geometry.log" 2>&1 &
pids+=("$!"); names+=(geometry)
CUDA_VISIBLE_DEVICES=3 "$PY" grids/eval/final_v3_quality_gpu.py --out "$OUT" \
  --metric conjuration --shard-index "$SHARD" --shard-count "$SHARDS" \
  >"$LOG/conjuration.log" 2>&1 &
pids+=("$!"); names+=(conjuration)

rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || { echo "FAILED ${names[$i]}"; rc=1; }
done
if [ "$rc" -ne 0 ]; then touch "$LOG/FAILED"; exit 1; fi
touch "$LOG/COMPLETE"
echo "FRODO8_YUME_EVAL_SHARD_COMPLETE shard=$SHARD/$SHARDS $(date -Is)"
