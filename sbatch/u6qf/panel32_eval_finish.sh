#!/bin/bash
# Merge, score, and fail-closed validate the complete mixed panel32 run.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/panel32_eval_env.sh"
CONJURATION_V2_FINAL=${PANEL32_CONJURATION_V2_FINAL:-$OUT/conjuration_v2_final}
CONJURATION_V2_WINDOWS="$CONJURATION_V2_FINAL/conjuration_v2_windows.csv"
CONJURATION_V2_EVENTS="$CONJURATION_V2_FINAL/conjuration_v2_events_adjudicated.csv"
rm -f "$OUT/logs/EVALUATION_COMPLETE"
test -f "$OUT/logs/SETUP_COMPLETE"
test -f "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE"
read -r SHARDS EXPECTED_SCENES EXPECTED_MODELS < <("$PY" - "$OUT/logs/shard_plan.json" <<'PY'
import json, sys
plan = json.load(open(sys.argv[1]))
print(int(plan["shard_count"]), int(plan["contexts"]) * 9, int(plan["models"]))
PY
)
for task in cpu style control geometry conjuration longreloc; do
  for ((shard=0; shard<SHARDS; shard++)); do
    test -f "$OUT/markers/${task}_shard${shard}.COMPLETE"
    test ! -f "$OUT/markers/${task}_shard${shard}.FAILED"
  done
done

FINISH_WORKERS=${PANEL32_FINISH_WORKERS:-32}
case "$FINISH_WORKERS" in
  *[!0-9]*|'') echo "invalid PANEL32_FINISH_WORKERS=$FINISH_WORKERS" >&2; exit 2 ;;
esac
if [ "$FINISH_WORKERS" -lt 1 ]; then
  echo "PANEL32_FINISH_WORKERS must be positive" >&2
  exit 2
fi
export OMP_NUM_THREADS="$FINISH_WORKERS"
export OPENBLAS_NUM_THREADS="$FINISH_WORKERS"
cd "$CODE"
"$PY" grids/eval/final_v2_summarize.py --out "$OUT" \
  >"$OUT/logs/cpu_summary.log" 2>&1
if [ "${PANEL32_PREMERGED_RELOCATION:-0}" = 1 ]; then
  test -s "$OUT/relocation_panel_rows.csv"
else
  "$PY" grids/eval/final_v2_relocation.py --out "$OUT" \
    >"$OUT/logs/relocation.log" 2>&1
fi
"$PY" grids/eval/iclr_long_relocation.py --out "$OUT/long_relocation" \
  --merge --expected-scenes "$EXPECTED_SCENES" --expected-models "$EXPECTED_MODELS" \
  >"$OUT/logs/long_relocation_merge.log" 2>&1
if [ "${PANEL32_PREMERGED_DENSE:-0}" = 1 ]; then
  test -s "$OUT/long_relocation/long_relocation_dense.csv"
else
  "$PY" grids/eval/iclr_dense_relocation_gate.py \
    --events "$OUT/long_relocation/long_relocation_events.csv" \
    --manifest "$OUT/video_manifest.csv" \
    --out "$OUT/long_relocation/long_relocation_dense.csv" --workers "$FINISH_WORKERS" \
    >"$OUT/logs/long_relocation_dense.log" 2>&1
fi
ABRUPT=$("$PY" - "$OUT/long_relocation/long_relocation_dense.csv" <<'PY'
import pandas as pd, sys
d = pd.read_csv(sys.argv[1])
print(int(d.abrupt_cut.sum()) if len(d) else 0)
PY
)
ADJUDICATION_ARGS=()
if [ "$ABRUPT" -gt 0 ]; then
  "$PY" grids/eval/render_iclr_relocation_candidates.py \
    --events "$OUT/long_relocation/long_relocation_dense.csv" \
    --manifest "$OUT/video_manifest.csv" \
    --out "$OUT/long_relocation/audit/candidates" --abrupt-only \
    >"$OUT/logs/long_relocation_candidates.log" 2>&1
  if [ ! -s "$OUT/long_relocation_adjudication.csv" ]; then
    echo "HUMAN_ADJUDICATION_REQUIRED abrupt_candidates=$ABRUPT file=$OUT/long_relocation_adjudication.csv" >&2
    exit 76
  fi
  ADJUDICATION_ARGS=(--adjudication "$OUT/long_relocation_adjudication.csv")
fi
"$PY" grids/eval/finalize_iclr_long_relocation.py \
  --direct "$OUT/relocation_panel_rows.csv" \
  --chain "$OUT/long_relocation/long_relocation_rows.csv" \
  --events "$OUT/long_relocation/long_relocation_events.csv" \
  --dense "$OUT/long_relocation/long_relocation_dense.csv" \
  "${ADJUDICATION_ARGS[@]}" --out "$OUT" \
  >"$OUT/logs/long_relocation_finalize.log" 2>&1
"$PY" grids/eval/final_v3_quality_summary.py --out "$OUT" \
  >"$OUT/logs/quality_summary.log" 2>&1
"$PY" grids/eval/validate_action_conventions.py --out "$OUT" \
  --require-positive minwm,minwm_ode,yume5b \
  >"$OUT/logs/action_conventions.log" 2>&1
"$PY" grids/eval/final_v3_validate.py --out "$OUT" \
  >"$OUT/logs/final_validation.log" 2>&1
# The paper observations must come from the completed human-adjudicated v2
# audit.  Never fall back to the legacy detector merely because finalization
# has not happened yet.
if [ ! -s "$CONJURATION_V2_WINDOWS" ] || [ ! -s "$CONJURATION_V2_EVENTS" ]; then
  echo "HUMAN_ADJUDICATION_REQUIRED conjuration_v2_final=$CONJURATION_V2_FINAL" >&2
  exit 76
fi
if [ ! -s "$OUT/long_relocation_events_adjudicated.csv" ]; then
  echo "MISSING_FINAL_LONG_RELOCATION_ADJUDICATION file=$OUT/long_relocation_events_adjudicated.csv" >&2
  exit 76
fi
"$PY" grids/eval/iclr_metric_trajectories.py --out "$OUT" \
  --figures "$OUT/figures" >"$OUT/logs/figures.log" 2>&1
"$PY" grids/eval/render_iclr_rollout_samples.py \
  --manifest "$OUT/video_manifest.csv" --figures "$OUT/figures" \
  --eval-config "$CONFIG" >"$OUT/logs/rollout_figures.log" 2>&1
for group in main ode dmd; do
  config="$CODE/grids/eval/panel32_aggregate_${group}.json"
  observations="$OUT/panel32_${group}_observations.csv"
  "$PY" grids/eval/panel32_convert_results.py \
    --manifest "$LOCKED_MANIFEST" \
    --config "$config" --evaluation-out "$OUT" --output "$observations" \
    --finalized-conjuration-v2-windows "$CONJURATION_V2_WINDOWS" \
    >"$OUT/logs/normalize_${group}.log" 2>&1
  "$PY" grids/eval/panel32_aggregate.py aggregate \
    --manifest "$LOCKED_MANIFEST" \
    --config "$config" --observations "$observations" \
    --output "$OUT/panel32_${group}_aggregate" \
    >"$OUT/logs/aggregate_${group}.log" 2>&1
done
"$PY" grids/eval/panel32_paper_report.py \
  --manifest "$LOCKED_MANIFEST" \
  --main-config "$CODE/grids/eval/panel32_aggregate_main.json" \
  --main-observations "$OUT/panel32_main_observations.csv" \
  --ode-config "$CODE/grids/eval/panel32_aggregate_ode.json" \
  --ode-observations "$OUT/panel32_ode_observations.csv" \
  --dmd-config "$CODE/grids/eval/panel32_aggregate_dmd.json" \
  --dmd-observations "$OUT/panel32_dmd_observations.csv" \
  --long-relocation-adjudication "$OUT/long_relocation_events_adjudicated.csv" \
  --long-conjuration-adjudication "$CONJURATION_V2_EVENTS" \
  --output-json "$OUT/panel32_final_report.json" \
  --output-markdown "$OUT/panel32_final_report.md" \
  >"$OUT/logs/paper_report.log" 2>&1
touch "$OUT/logs/EVALUATION_COMPLETE"
echo "PANEL32_EVALUATION_COMPLETE $(date -Is)"
