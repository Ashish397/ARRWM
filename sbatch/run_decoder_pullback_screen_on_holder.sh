#!/bin/bash
# Full one-rotation decoder-pullback screen in an existing durable holder.
set -euo pipefail
: "${ALLOC:?set ALLOC}"
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
DATA=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination
FIELD=$DATA/surrogate_field_2708
CAPTURE=$DATA/surrogate_capture_2708
OUT=${OUT:-$FIELD/decoder_pullback_r0_h${ALLOC}}
ROTATION=${ROTATION:-0}
VGG_VECTORS_PER_ROLE=${VGG_VECTORS_PER_ROLE:-1}
UPDATES=${UPDATES:-600}
BATCH_SIZE=${BATCH_SIZE:-2}

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

BANK="$OUT/vgg_r${ROTATION}_live_pullback.npz"
CUDA_VISIBLE_DEVICES=0 python \
  analysis/gan_tuning/decoder_pullback_benchmark.py extract \
  --capture-dir "$CAPTURE" \
  --head "$FIELD/head_curves/vgg_r${ROTATION}_lr1e3.pt" \
  --rotation "$ROTATION" --crop-index 0 --output "$BANK" \
  --vgg-vectors-per-role "$VGG_VECTORS_PER_ROLE" \
  > "$OUT/extract.log" 2>&1

pids=()
gpu=0
for arm in conditioned z_only shuffled; do
  CUDA_VISIBLE_DEVICES="$gpu" python \
    analysis/gan_tuning/decoder_pullback_benchmark.py fit \
    --bank "$BANK" --output "$OUT/results/${arm}.json" --arm "$arm" \
    --updates "$UPDATES" --checkpoints "0,1,3,10,30,100,300,600,$UPDATES" \
    --seeds 3 --batch-size "$BATCH_SIZE" --eval-batch-size 2 \
    --width 96 --blocks 4 \
    > "$OUT/results/${arm}.log" 2>&1 &
  pids+=("$!")
  gpu=$((gpu + 1))
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 120 "$OUT"/results/*.log || true
  exit 61
fi

python - <<PY | tee "$OUT/SUMMARY.txt"
import json
from pathlib import Path
out = Path("$OUT")
for arm in ("conditioned", "z_only", "shuffled"):
    d = json.load(open(out / "results" / f"{arm}.json"))
    print("ARM", arm, "parameters", d["parameters"],
          "presentations", d["target_presentations"])
    for key in (
        "unseen_z_unseen_v",
        "unseen_z_unseen_v/fixed_vgg_head/positive",
        "unseen_z_unseen_v/structured_multiscale/positive",
    ):
        per_seed = [r["final"][key]["cosine"] for r in d["results"]]
        print(key,
              "worst_seed_q1", min(v["q1"] for v in per_seed),
              "worst_seed_median", min(v["median"] for v in per_seed),
              "seeds", [(v["q1"], v["median"]) for v in per_seed])
    print("curve_worst_q1", {
        step: min(
            r["curve_unseen_z_unseen_v"][step]["cosine"]["q1"]
            for r in d["results"]
        )
        for step in d["results"][0]["curve_unseen_z_unseen_v"]
    })
    print("max_structural_error", max(
        max(r["structural"].values()) for r in d["results"]
    ))
PY

echo "[DECODER-PULLBACK] r${ROTATION} screen complete vgg_vectors_per_role=$VGG_VECTORS_PER_ROLE output=$OUT"
