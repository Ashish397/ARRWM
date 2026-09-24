#!/bin/bash
# Run shard zero, wait for peer shards, merge them, validate and summarize.
set -euo pipefail
R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
CODE="$R/ARRWM"
STAGE="$R/minwm_seed29/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
OUT="$CODE/grids/eval/out_minwm_seed29_sensitivity_20260920"
OLD="$CODE/grids/eval/out_iclr_final_v3_full_20260917"
export ARR="$STAGE"
export MINWM_SEED29_DIR="$STAGE/logs/eval_final/fleet30s/minwm_seed29"
export SEED65_DIR="$STAGE/analysis/eval_final/seed65_e1"

EVAL_SHARD=0 "$R/minwm_seed29_eval_worker.sh"

deadline=$((SECONDS + 21600))
while [ "$SECONDS" -lt "$deadline" ]; do
  complete=0
  failed=0
  for shard in 0 1 2 3; do
    [ -f "$OUT/logs/shard${shard}/COMPLETE" ] && complete=$((complete+1))
    [ -f "$OUT/logs/shard${shard}/FAILED" ] && failed=$((failed+1))
  done
  echo "WAIT_EVAL_SHARDS complete=$complete failed=$failed $(date -Is)"
  [ "$failed" -eq 0 ] || exit 1
  if [ "$complete" -eq 4 ]; then
    break
  fi
  sleep 20
done
[ "$complete" -eq 4 ] || { echo "timed out waiting for evaluation shards" >&2; exit 1; }

"$PY" - "$OUT" <<'PY'
import sys
from pathlib import Path
import pandas as pd
out = Path(sys.argv[1])
spec = {
    'cpu_endpoints': (3 * 288, ['scene', 'model', 'horizon_s']),
    'cpu_windows': (6 * 288, ['scene', 'model', 'window_start_s']),
    'style_windows': (6 * 288, ['scene', 'model', 'window_start_s']),
    'control_windows': (6 * 288, ['scene', 'model', 'window_start_s']),
}
for stem, (expected, keys) in spec.items():
    parts = [pd.read_csv(out / f'{stem}_shard{i}.csv') for i in range(4)]
    data = pd.concat(parts, ignore_index=True).sort_values(keys)
    assert len(data) == expected, (stem, len(data), expected)
    assert not data.duplicated(keys).any(), stem
    data.to_csv(out / f'{stem}.csv', index=False)
for metric in ('geometry', 'conjuration'):
    files = list((out / metric).glob('*.json'))
    assert len(files) == 288, (metric, len(files))
print('SHARDS_MERGED_AND_VALIDATED')
PY

"$PY" "$CODE/grids/eval/minwm_seed29_sensitivity.py" summarize \
  --out "$OUT" --old "$OLD" >"$OUT/logs/summarize.log" 2>&1
echo "MINWM_SEED29_EVAL_COMPLETE $(date -Is)" | tee "$OUT/logs/status.log"
