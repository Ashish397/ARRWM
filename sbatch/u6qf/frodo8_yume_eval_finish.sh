#!/bin/bash
# Merge, aggregate, and fail-closed validate the isolated Frodo8 pilot.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/frodo8_yume_eval_env.sh"
rm -f "$OUT/logs/EVALUATION_COMPLETE"
test -f "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE"
SHARDS=$("$PY" - "$OUT/logs/shard_plan.json" <<'PY'
import json, sys
print(int(json.load(open(sys.argv[1]))["shard_count"]))
PY
)
for ((shard=0; shard<SHARDS; shard++)); do
  test -f "$OUT/logs/shard${shard}/COMPLETE"
done
export OMP_NUM_THREADS=32
cd "$CODE"

"$PY" grids/eval/final_v2_summarize.py --out "$OUT" >"$OUT/logs/cpu_summary.log" 2>&1
"$PY" grids/eval/final_v2_relocation.py --out "$OUT" >"$OUT/logs/relocation.log" 2>&1
"$PY" grids/eval/final_v3_quality_summary.py --out "$OUT" >"$OUT/logs/quality_summary.log" 2>&1
# The eight-context smoke is too small to require a strictly positive minWM
# throttle delta: a low response is a measured model outcome, whereas a
# negative delta would still fail validate_action_conventions.py's unconditional
# reversed-axis gate.  minWM's common-yaw adapter and sidecars are checked
# directly below, and the independent action preflight has already passed.
"$PY" grids/eval/validate_action_conventions.py --out "$OUT" \
  --require-positive minwm_ode,yume5b >"$OUT/logs/action_conventions.log" 2>&1
"$PY" grids/eval/final_v3_validate.py --out "$OUT" >"$OUT/logs/final_validation.log" 2>&1
touch "$OUT/logs/EVALUATION_COMPLETE"
echo "FRODO8_YUME_EVALUATION_COMPLETE $(date -Is)"
