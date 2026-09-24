#!/bin/bash
set -euo pipefail
: "${RELOC_SHARD:?set RELOC_SHARD}"
: "${RELOC_SHARDS:=4}"
R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
CODE="$R/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
OUT="$CODE/grids/eval/out_iclr_relocation_20260920"
MANIFEST="$OUT/video_manifest.csv"
SEEDS="$R/eval30s_check/ARR/analysis/eval_final/seed65_e1"
mkdir -p "$OUT/logs"
cd "$CODE"
"$PY" grids/eval/iclr_relocation.py \
  --manifest "$MANIFEST" --out "$OUT" --seed-dir "$SEEDS" \
  --remote-prefix "$R/eval30s_check/ARR" \
  --shard-index "$RELOC_SHARD" --shard-count "$RELOC_SHARDS" --workers 6 \
  >"$OUT/logs/shard${RELOC_SHARD}.log" 2>&1
touch "$OUT/logs/shard${RELOC_SHARD}.complete"
