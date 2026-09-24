#!/bin/bash
# Validate all 4,320 generated videos, build the immutable metric manifest,
# decode-audit every clip, and verify the exact conditioning boundary.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/panel32_eval_env.sh"
SHARDS=${PANEL32_EVAL_SHARDS:-48}
case "$SHARDS" in *[!0-9]*|'') echo "invalid PANEL32_EVAL_SHARDS=$SHARDS" >&2; exit 2 ;; esac
if [ "$SHARDS" -lt 4 ] || [ $((SHARDS % 4)) -ne 0 ]; then
  echo "PANEL32_EVAL_SHARDS must be a positive multiple of four and at least four" >&2
  exit 2
fi

test -s "$CONFIG"
mkdir -p "$OUT/logs" "$OUT/markers" "$OUT/preflight"
if [ -s "$OUT/logs/shard_plan.json" ]; then
  "$PY" - "$OUT/logs/shard_plan.json" "$SHARDS" "$CONFIG" "$PANEL32_MODELS" <<'PY'
import hashlib, json, sys
plan_path, requested_shards, config_path, model_csv = sys.argv[1:]
plan = json.load(open(plan_path))
old = int(plan["shard_count"])
new = int(requested_shards)
assert old == new, (
    f"refusing to mix shard plans: existing={old}, requested={new}; "
    "use the existing count or a new PANEL32_EVAL_OUT"
)
models = model_csv.split(",")
if "model_ids" in plan:
    assert plan["model_ids"] == models, (
        "refusing to mix model sets in one evaluation output; use a new "
        "PANEL32_EVAL_OUT"
    )
if "config_sha256" in plan:
    digest = hashlib.sha256(open(config_path, "rb").read()).hexdigest()
    assert plan["config_sha256"] == digest, (
        "refusing to mix evaluation configs in one output; use a new "
        "PANEL32_EVAL_OUT"
    )
PY
fi
rm -f "$OUT/logs/SETUP_COMPLETE" "$OUT/logs/EVALUATION_COMPLETE" \
  "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE"
rm -f "$OUT/preflight"/action_preflight_rows_shard*.csv \
  "$OUT/preflight/action_preflight_rows.csv" \
  "$OUT/preflight/action_preflight_audit.csv" \
  "$OUT/markers"/preflight_shard*.COMPLETE \
  "$OUT/markers"/preflight_shard*.FAILED
find "$OUT/markers" -maxdepth 1 -type f \( -name '*.COMPLETE' -o -name '*.FAILED' \) -delete

cd "$CODE"
reuse_audits=0
if [ -s "$OUT/generation_validation.json" ] && \
   [ -s "$OUT/video_manifest.csv" ] && \
   [ -s "$OUT/context_alignment_rows.csv" ] && \
   [ -s "$OUT/logs/context_alignment.log" ] && \
   "$PY" - "$OUT/generation_validation.json" "$OUT/video_manifest.csv" \
      "$OUT/context_alignment_rows.csv" "$OUT/logs/context_alignment.log" \
      "$CONFIG" "$PANEL32_MODELS" \
      "$CODE/grids/eval/panel32_validate_generations.py" <<'PY'
import csv
import json
import sys
from pathlib import Path

report_path, manifest_path, alignment_path, alignment_log_path = map(Path, sys.argv[1:5])
config_path = Path(sys.argv[5])
models = sys.argv[6].split(",")
validator_path = Path(sys.argv[7])
config = json.loads(config_path.read_text())
report = json.loads(report_path.read_text())
expected = len(config["context_ids"]) * 9 * len(models)
assert report["status"] == "pass"
assert report["expected_videos"] == expected
assert report["validated_videos"] == expected
assert report["action_invariant_seed_groups"] == len(config["context_ids"]) * len(models)
assert not report["errors"] and len(report["rows"]) == expected
assert set(report["models"]) == set(models) == set(config["models"])

# Reuse is permitted only when neither code/config nor any audited video or
# sidecar changed after the atomic validation report was written.
report_time = report_path.stat().st_mtime_ns
assert report_time >= config_path.stat().st_mtime_ns
assert report_time >= validator_path.stat().st_mtime_ns
for row in report["rows"]:
    for key in ("path", "sidecar"):
        path = Path(row[key])
        assert path.is_file() and path.stat().st_mtime_ns <= report_time

for path in (manifest_path, alignment_path, alignment_log_path):
    assert path.is_file() and path.stat().st_mtime_ns >= report_time
with manifest_path.open(newline="") as stream:
    manifest_rows = list(csv.DictReader(stream))
assert len(manifest_rows) == expected
assert set(row["model"] for row in manifest_rows) == set(models)
with alignment_path.open(newline="") as stream:
    alignment_rows = list(csv.DictReader(stream))
assert len(alignment_rows) == expected
assert set(row["model"] for row in alignment_rows) == set(models)
alignment = json.loads(alignment_log_path.read_text())
assert alignment["status"] == "pass"
assert alignment["videos"] == expected
assert alignment["models"] == len(models)
assert alignment["common_generation_boundary_real_frame"] == 32
print(f"PANEL32_SETUP_AUDITS_FRESH videos={expected} models={len(models)}")
PY
then
  reuse_audits=1
fi
if [ "$reuse_audits" -eq 0 ]; then
  "$PY" grids/eval/panel32_validate_generations.py \
    --config "$CONFIG" \
    --report "$OUT/generation_validation.json" \
    >"$OUT/logs/generation_validation.log" 2>&1
  # Keep a compact, model-specific attestation for the released Matrix-Game
  # execution contract.  This catches accidental mixing of legacy seed-1234
  # outputs with the released-default seed-0 universal runner even when the
  # media itself decodes and aligns correctly.
  "$PY" grids/eval/panel32_validate_generations.py \
    --config "$CONFIG" --models matrixgame2 \
    --report "$OUT/matrixgame_strict_validation.json" \
    >"$OUT/logs/matrixgame_strict_validation.log" 2>&1
  "$PY" grids/eval/final_v2_iclr.py --out "$OUT" --stage manifest \
    >"$OUT/logs/manifest.log" 2>&1
  "$PY" grids/eval/final_v2_iclr.py --out "$OUT" --stage decode \
    >"$OUT/logs/decode.log" 2>&1
  "$PY" grids/eval/final_v2_iclr.py --out "$OUT" --stage enrich \
    >"$OUT/logs/enrich.log" 2>&1
  "$PY" grids/eval/validate_manifest_context_alignment.py \
    --manifest "$OUT/video_manifest.csv" \
    --eval-config "$CONFIG" \
    --workers 16 \
    --output "$OUT/context_alignment_rows.csv" \
    >"$OUT/logs/context_alignment.log" 2>&1
else
  echo "PANEL32_SETUP_REUSING_FRESH_PASSED_AUDITS" | tee -a "$OUT/logs/setup_resume.log"
fi

"$PY" - "$OUT/video_manifest.csv" "$CONFIG" "$PANEL32_MODELS" \
  "$LOCKED_MANIFEST" "$CODE/grids/eval" <<'PY'
import hashlib
import json
import sys
import pandas as pd

manifest_path, config_path, model_csv, locked_path, aggregate_dir = sys.argv[1:]
config = json.load(open(config_path))
models = model_csv.split(",")
contexts = list(config["context_ids"])
actions = 9
locked_bytes = open(locked_path, "rb").read()
locked = json.loads(locked_bytes)
locked_sha = hashlib.sha256(locked_bytes).hexdigest()
assert config["panel_manifest_sha256"] == locked_sha
assert config["panel_id"] == locked["panel_id"]
assert contexts == [row["context_id"] for row in locked["contexts"]]
expected_seats = {
    "ours": "ours_no_gan", "lingbot": "lingbot",
    "dreamx": "dreamx", "matrixgame2": "matrixgame2",
    "minwm": "minwm", "yume5b": "yume5b",
}
assert config["family_seats"] == expected_seats
for suffix, config_key in {
        "main": "main_models", "ode": "ode_models",
        "dmd": "ablation_models"}.items():
    aggregate = json.load(open(
        f"{aggregate_dir}/panel32_aggregate_{suffix}.json"))
    report_models = [row["model_id"] for row in aggregate["models"]]
    assert aggregate["family_seat_count"] == 6
    assert report_models == config[config_key], (
        suffix, report_models, config[config_key])
data = pd.read_csv(manifest_path)
assert len(contexts) == 32 and len(set(contexts)) == len(contexts)
assert len(config["models"]) == len(models)
assert set(config["models"]) == set(models)
assert data.uid.nunique() == len(contexts) and set(data.uid) == set(contexts)
assert data.model.nunique() == len(models) and set(data.model) == set(models)
assert len(data) == len(contexts) * actions * len(models)
assert (data.groupby("model").size() == len(contexts) * actions).all()
assert data.local_video.all() and data.decoded_frames.notna().all()
assert (data.decoded_frames == data.container_frames).all()
assert (data.generated_duration_s >= 30).all()
for model, record in config["models"].items():
    assert set(data.loc[data.model == model, "context_frames"]) == {record["context_frames"]}
print(f"PANEL32_MANIFEST_VALIDATED rows={len(data)} models={len(models)} contexts={len(contexts)}")
PY

"$PY" - "$OUT/logs/shard_plan.json" "$SHARDS" "$CONFIG" "$PANEL32_MODELS" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

path, shards, config_path, model_csv = (
    Path(sys.argv[1]), int(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
)
config = json.loads(config_path.read_text())
models = model_csv.split(",")
contexts = len(config["context_ids"])
assert set(config["models"]) == set(models)
payload = {
    "schema_version": 1,
    "shard_count": shards,
    "shards_per_holder": 4,
    "batch_count": shards // 4,
    "contexts": contexts,
    "models": len(models),
    "model_ids": models,
    "videos": contexts * 9 * len(models),
    "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    "tasks": ["cpu", "style", "control", "geometry", "conjuration", "longreloc"],
}
temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
temporary.replace(path)
PY
touch "$OUT/logs/SETUP_COMPLETE"
echo "PANEL32_EVAL_SETUP_COMPLETE shards=$SHARDS $(date -Is)"
