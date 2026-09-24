#!/bin/bash
# Strict whole-fleet validation for one FrodoBots-eight model row.
set -euo pipefail

MODEL=${1:?usage: frodo8_validate_model.sh MODEL}
case "$MODEL" in
  lingbot|dreamx|matrixgame2|minwm|minwm_ode) ;;
  *) echo "unsupported Frodo8 model: $MODEL" >&2; exit 64 ;;
esac

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
S=${ALIGNED32_STAGE:-$R/aligned32_stage}
PY=${ARRWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python}
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"

case "$MODEL" in
  minwm_ode) OUT=$S/minwm_ode ;;
  *) OUT=$S/fleet30s_aligned32/$MODEL ;;
esac

mkdir -p "$S/reports/frodo8_generation"
exec "$PY" "$A/grids/eval/frodo8_generation.py" validate \
  --manifest "$A/experiments/e1/scene_shortlist/frodo8_manifest.json" \
  --source-manifest "$A/experiments/e1/scene_shortlist/e1_32_windows.json" \
  --model "$MODEL" --output "$OUT" \
  --seed-frame32 "$S/seed_frame32" \
  --seed-stream "$S/seed_stream_aligned" \
  --report "$S/reports/frodo8_generation/${MODEL}.json"
