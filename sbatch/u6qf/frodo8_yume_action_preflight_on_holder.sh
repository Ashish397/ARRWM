#!/bin/bash
# Run on one GPU inside an existing holder after setup.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/frodo8_yume_eval_env.sh"
test -f "$OUT/logs/SETUP_COMPLETE"
mkdir -p "$OUT/preflight"
rm -f "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE"
require_four_distinct_cuda_ordinals action_preflight
cd "$CODE"
CUDA_VISIBLE_DEVICES=0 "$PY" \
  grids/eval/preflight_action_conventions.py \
  --out "$OUT/preflight" \
  --uids u31,u04,a20,m30,m38,m89,m128,b36 \
  --models lingbot,dreamx,minwm,matrixgame2,yume5b,minwm_ode \
  --require-positive minwm,minwm_ode,yume5b
echo "FRODO8_YUME_ACTION_PREFLIGHT_COMPLETE $(date -Is)"
