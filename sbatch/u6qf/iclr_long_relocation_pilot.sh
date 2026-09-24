#!/bin/bash
set -euo pipefail
R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
CODE="$R/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
SOURCE="$CODE/grids/eval/out_iclr_relocation_20260920"
OUT="$CODE/grids/eval/out_iclr_long_relocation_pilot_20260920"
mkdir -p "$OUT/logs"
cd "$CODE"
"$PY" grids/eval/iclr_long_relocation.py \
  --manifest "$SOURCE/video_manifest.csv" --out "$OUT" \
  --remote-prefix "$R/eval30s_check/ARR" \
  --scenes m33_R,m111_FR,m30_B,m83_FR,m36_R,m128_R,n12_N,u04_L \
  --workers 16 --force >"$OUT/logs/pilot.log" 2>&1
"$PY" grids/eval/iclr_long_relocation.py --out "$OUT" --merge --expected-scenes 8 \
  >>"$OUT/logs/pilot.log" 2>&1
touch "$OUT/logs/pilot.complete"
