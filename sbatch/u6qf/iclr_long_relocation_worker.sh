#!/bin/bash
set -euo pipefail
: "${LONG_RELOC_SHARD:?set LONG_RELOC_SHARD}"
: "${LONG_RELOC_SHARDS:=4}"
R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
CODE="$R/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
SOURCE="$CODE/grids/eval/out_iclr_relocation_20260920"
OUT="$CODE/grids/eval/out_iclr_long_relocation_20260920"
mkdir -p "$OUT/logs"
cd "$CODE"
"$PY" grids/eval/iclr_long_relocation.py \
  --manifest "$SOURCE/video_manifest.csv" --out "$OUT" \
  --remote-prefix "$R/eval30s_check/ARR" \
  --shard-index "$LONG_RELOC_SHARD" --shard-count "$LONG_RELOC_SHARDS" \
  --workers 8 >"$OUT/logs/shard${LONG_RELOC_SHARD}.log" 2>&1
touch "$OUT/logs/shard${LONG_RELOC_SHARD}.complete"
