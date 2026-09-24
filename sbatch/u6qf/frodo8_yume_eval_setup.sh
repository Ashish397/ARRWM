#!/bin/bash
# CPU gate and immutable 8-context pilot manifest. Run inside any holder/login
# allocation only after all six external fleets (including YUME) are complete.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/frodo8_yume_eval_env.sh"
mkdir -p "$OUT/logs"
rm -f "$OUT/logs/SETUP_COMPLETE" "$OUT/logs/EVALUATION_COMPLETE" \
  "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE" \
  "$OUT/logs"/shard*/COMPLETE "$OUT/logs"/shard*/FAILED

SHARDS=${FRODO8_EVAL_SHARDS:-4}
case "$SHARDS" in *[!0-9]*|'') echo "invalid FRODO8_EVAL_SHARDS=$SHARDS" >&2; exit 2 ;; esac
if [ "$SHARDS" -lt 1 ]; then
  echo "FRODO8_EVAL_SHARDS must be positive" >&2
  exit 2
fi

cd "$CODE"
"$PY" grids/eval/validate_aligned32_fleets.py \
  --root "$ALIGNED" \
  --dreamx-dir "$DREAMX30S_DIR" \
  --models lingbot,dreamx,minwm,matrixgame2,yume5b,minwm_ode \
  --uids "$UID_JSON" \
  --seed-frame32 "$R/aligned32_stage/seed_frame32" \
  --seed65-dir "$SEED65_DIR" \
  --minwm-ode-dir "$MINWM_ODE_DIR" \
  --matrix-runner "$CODE/code_release/baselines/matrixgame_runner.py" \
  --matrix-config "$R/aligned32_stage/Matrix-Game-2/configs/inference_yaml/inference_universal.yaml" \
  --matrix-checkpoint "$R/aligned32_stage/Matrix-Game-2/Matrix-Game-2.0/base_distilled_model/base_distill.safetensors" \
  --yume-dir "$YUME" \
  --yume-runner "$CODE/code_release/baselines/yume5b_frodo8_runner.py" \
  --yume-vendor-entrypoint "$CODE/third_party/YUME/fastvideo/sample/sample_5b.py" \
  --yume-checkpoint "$R/yume/Yume-5B-720P/diffusion_pytorch_model.safetensors" \
  >"$OUT/logs/fleet_validation.json"

"$PY" grids/eval/final_v2_iclr.py --out "$OUT" --stage manifest \
  >"$OUT/logs/manifest.log" 2>&1
"$PY" grids/eval/final_v2_iclr.py --out "$OUT" --stage decode \
  >"$OUT/logs/decode.log" 2>&1
"$PY" grids/eval/final_v2_iclr.py --out "$OUT" --stage enrich \
  >"$OUT/logs/enrich.log" 2>&1
"$PY" grids/eval/validate_manifest_context_alignment.py \
  --manifest "$OUT/video_manifest.csv" \
  --seed65-dir "$SEED65_DIR" \
  --workers 4 \
  --output "$OUT/context_alignment_rows.csv" \
  >"$OUT/logs/context_alignment.json"

"$PY" - "$OUT/video_manifest.csv" "$FRODO8_UIDS" "$FRODO8_MODELS" <<'PY'
import sys
import pandas as pd

path, uid_csv, model_csv = sys.argv[1:]
uids = uid_csv.split(",")
models = model_csv.split(",")
data = pd.read_csv(path)
assert set(data.uid) == set(uids) and data.uid.nunique() == 8
assert set(data.model) == set(models) and data.model.nunique() == 16
assert len(data) == 8 * 9 * 16
assert (data.groupby("model").size() == 72).all()
assert data.local_video.all() and data.decoded_frames.notna().all()
assert (data.decoded_frames == data.container_frames).all()
assert (data.generated_duration_s >= 30).all()
assert set(data.loc[data.model == "yume5b", "context_frames"]) == {1}
print("FRODO8_YUME_MANIFEST_VALIDATED rows=1152 models=16 contexts=8")
PY

cat >"$OUT/logs/shard_plan.json" <<EOF
{"shard_count": $SHARDS, "contexts": 8, "models": 16, "videos": 1152}
EOF
touch "$OUT/logs/SETUP_COMPLETE"
echo "FRODO8_YUME_EVAL_SETUP_COMPLETE $(date -Is)"
