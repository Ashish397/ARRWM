#!/bin/bash
# Second gated geometry pilot inside the still-running u6qf holder.
set -u
BASE=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
ROOT="$BASE/eval30s_check/ARR"
PY="$BASE/miniforge3/envs/arrwm/bin/python3.10"
OUT="$ROOT/grids/eval/out_dual_reference_pilot"
MANIFEST="$ROOT/grids/eval/v4_manifest_u6qf.csv"
export HF_HOME="$BASE/iclrv3_hf_cache"
export TORCH_HOME="$BASE/torch_home"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd "$ROOT"

CUDA_VISIBLE_DEVICES=1 "$PY" grids/eval/final_v4_dual_reference.py \
  --manifest "$MANIFEST" --out "$OUT/geometry_pilot_v2.csv" \
  --metric geometry --pilot > "$OUT/geometry_pilot_v2.log" 2>&1
run_status=$?
if [ "$run_status" -eq 0 ]; then
  "$PY" grids/eval/validate_dual_reference_pilot.py \
    --style "$OUT/style_pilot.csv" --geometry "$OUT/geometry_pilot_v2.csv" \
    --report "$OUT/validation_v2.json" > "$OUT/validation_v2.log" 2>&1
  run_status=$?
fi
tail -10 "$OUT/geometry_pilot_v2.log" 2>/dev/null || true
cat "$OUT/validation_v2.json" 2>/dev/null || true
echo "dual_reference_pilot_v2_exit=$run_status"
exit "$run_status"
