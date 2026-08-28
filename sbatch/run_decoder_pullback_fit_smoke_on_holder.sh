#!/bin/bash
# Actual-geometry/architecture fit smoke using the passed identity bank.
set -euo pipefail
: "${ALLOC:?set ALLOC}"
: "${BANK:?set BANK}"
: "${OUT:?set OUT}"

cd /scratch/u6ex/as1748.u6ex/ARRWM
mkdir -p "$OUT"
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

pids=()
for spec in "conditioned 1" "z_only 2"; do
  read -r arm gpu <<< "$spec"
  CUDA_VISIBLE_DEVICES="$gpu" python \
    analysis/gan_tuning/decoder_pullback_benchmark.py fit \
    --bank "$BANK" --output "$OUT/${arm}.json" --arm "$arm" \
    --updates 2 --checkpoints 0,1,2 --seeds 1 \
    --batch-size 1 --eval-batch-size 1 --width 96 --blocks 4 \
    > "$OUT/${arm}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 100 "$OUT"/*.log || true
  exit 51
fi
python - <<PY
import json
from pathlib import Path
out = Path("$OUT")
for arm in ("conditioned", "z_only"):
    value = json.load(open(out / f"{arm}.json"))
    assert value["parameters"] > 0
    assert value["target_presentations"] == 2
    assert value["quadrant_counts"] == {
        "seen_z_seen_v": 4, "unseen_z_seen_v": 4,
        "seen_z_unseen_v": 4, "unseen_z_unseen_v": 4,
    }
    structural = value["results"][0]["structural"]
    assert structural["zero_relative_rms"] == 0.0
    assert structural["oddness_relative_rms"] < 1e-4
    assert structural["additivity_relative_rms"] < 1e-3
    print(arm, value["parameters"], structural)
PY
echo "[DECODER-PULLBACK] fit smoke PASS output=$OUT"
