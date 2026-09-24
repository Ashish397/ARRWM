#!/bin/bash
# Audit the released Matrix-Game execution contract independently of the
# whole-fleet media/alignment gate.
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/panel32_eval_env.sh"
mkdir -p "$OUT/logs"
rm -f "$OUT/logs/MATRIXGAME_STRICT_VALIDATION_COMPLETE"

cd "$CODE"
"$PY" grids/eval/panel32_validate_generations.py \
  --config "$CONFIG" --models matrixgame2 \
  --report "$OUT/matrixgame_strict_validation.json" \
  >"$OUT/logs/matrixgame_strict_validation.log" 2>&1
"$PY" - "$OUT/matrixgame_strict_validation.json" <<'PY'
import json
import sys

report = json.load(open(sys.argv[1]))
assert report["status"] == "pass", report["errors"][:3]
assert report["expected_videos"] == 288
assert report["validated_videos"] == 288
assert report["action_invariant_seed_groups"] == 32
assert not report["errors"]
print("MATRIXGAME_STRICT_VALIDATION_PASS videos=288 seed_groups=32")
PY
touch "$OUT/logs/MATRIXGAME_STRICT_VALIDATION_COMPLETE"

