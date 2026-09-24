#!/bin/bash
# Publish the already validated balanced-panel metrics without rerunning any
# GPU detector or the expensive direct/dense relocation scans.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/panel32_eval_env.sh"
CONJURATION_V2_FINAL=${PANEL32_CONJURATION_V2_FINAL:-$OUT/conjuration_v2_final}
CONJURATION_V2_WINDOWS="$CONJURATION_V2_FINAL/conjuration_v2_windows.csv"
CONJURATION_V2_EVENTS="$CONJURATION_V2_FINAL/conjuration_v2_events_adjudicated.csv"

rm -f "$OUT/logs/EVALUATION_COMPLETE"
test -f "$OUT/logs/SETUP_COMPLETE"
test -f "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE"
test -s "$OUT/long_relocation_events_adjudicated.csv"
test -s "$CONJURATION_V2_WINDOWS"
test -s "$CONJURATION_V2_EVENTS"

export OMP_NUM_THREADS=32
export OPENBLAS_NUM_THREADS=32
cd "$CODE"

# Re-run the cheap fail-closed validations immediately before publication.
"$PY" grids/eval/validate_action_conventions.py --out "$OUT" \
  --require-positive minwm,minwm_ode,yume5b \
  >"$OUT/logs/action_conventions.log" 2>&1
"$PY" grids/eval/final_v3_validate.py --out "$OUT" \
  >"$OUT/logs/final_validation.log" 2>&1

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
