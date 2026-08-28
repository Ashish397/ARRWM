#!/bin/bash
# All-three-row z scaling test for the decoder-pullback operator.
set -euo pipefail
: "${ALLOC:?set ALLOC}"
: "${OUT:?set OUT}"
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
DATA=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination
FIELD=$DATA/surrogate_field_2708
CAPTURE=$DATA/surrogate_capture_2708

cd "$ROOT"
mkdir -p "$OUT/results"
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m py_compile \
  model/decoder_pullback_surrogate.py \
  analysis/gan_tuning/decoder_pullback_benchmark.py
python -m pytest -q testing/test_decoder_pullback_surrogate.py \
  > "$OUT/model_tests.log" 2>&1

BANK="$OUT/vgg_r0_live_pullback_z3.npz"
CUDA_VISIBLE_DEVICES=0 python \
  analysis/gan_tuning/decoder_pullback_benchmark.py extract \
  --capture-dir "$CAPTURE" --head "$FIELD/head_curves/vgg_r0_lr1e3.pt" \
  --rotation 0 --crop-index 0 --z-rows 0,1,2 \
  --vgg-vectors-per-role 4 --output "$BANK" \
  > "$OUT/extract.log" 2>&1

SPECS=(
  "conditioned_w96 conditioned 96 0"
  "z_only_w96 z_only 96 1"
  "shuffled_w96 shuffled 96 2"
  "conditioned_w192 conditioned 192 3"
)
pids=()
for spec in "${SPECS[@]}"; do
  read -r name arm width gpu <<< "$spec"
  CUDA_VISIBLE_DEVICES="$gpu" python \
    analysis/gan_tuning/decoder_pullback_benchmark.py fit \
    --bank "$BANK" --output "$OUT/results/${name}.json" --arm "$arm" \
    --updates 900 --checkpoints 0,1,3,10,30,100,300,600,900 --seeds 3 \
    --batch-size 4 --eval-batch-size 4 --width "$width" --blocks 4 \
    > "$OUT/results/${name}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 120 "$OUT"/results/*.log || true
  exit 81
fi

python - <<PY | tee "$OUT/SUMMARY.txt"
import json
from pathlib import Path
out = Path("$OUT")
for name in ("conditioned_w96", "z_only_w96", "shuffled_w96", "conditioned_w192"):
    d = json.load(open(out / "results" / f"{name}.json"))
    print("ARM", name, "params", d["parameters"],
          "presentations", d["target_presentations"],
          "quadrants", d["quadrant_counts"])
    for key in (
        "unseen_z_unseen_v",
        "unseen_z_unseen_v/fixed_vgg_head/positive",
        "unseen_z_unseen_v/structured_multiscale/positive",
    ):
        values = [r["final"][key]["cosine"] for r in d["results"]]
        print(key,
              "worst_q1", min(v["q1"] for v in values),
              "worst_median", min(v["median"] for v in values),
              "seeds", [(v["q1"], v["median"]) for v in values])
    print("curve", {
        step: min(
            r["curve_unseen_z_unseen_v"][step]["cosine"]["q1"]
            for r in d["results"]
        ) for step in d["results"][0]["curve_unseen_z_unseen_v"]
    })
PY
echo "[DECODER-PULLBACK] zscale screen complete output=$OUT"
