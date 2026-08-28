#!/bin/bash
# Naturally paired VGG pullback extraction and controlled fit on a live holder.
set -euo pipefail
: "${ALLOC:?set ALLOC to the live holder job id}"

ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
DATA=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination
FIELD=$DATA/surrogate_field_2708
CAPTURE=$DATA/surrogate_capture_2708
HEAD=$FIELD/head_curves/vgg_r0_lr1e3.pt
OUT=${OUT:-$FIELD/paired_decoder_pullback_r0_h${ALLOC}}
SHARDS=4
UPDATES=${UPDATES:-900}

cd "$ROOT"
mkdir -p "$OUT/extract_shards" "$OUT/results"
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

python -m py_compile \
  analysis/gan_tuning/paired_decoder_pullback_benchmark.py \
  testing/test_paired_decoder_pullback_benchmark.py
python -m pytest -q \
  testing/test_decoder_pullback_surrogate.py \
  testing/test_paired_decoder_pullback_benchmark.py \
  > "$OUT/tests.log" 2>&1

pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" python \
    analysis/gan_tuning/paired_decoder_pullback_benchmark.py extract-shard \
    --capture-dir "$CAPTURE" --head "$HEAD" --rotation 0 \
    --output "$OUT/extract_shards/shard_${shard}.npz" \
    --z-rows 0,1,2 --crop-index 0 \
    --shard-index "$shard" --num-shards "$SHARDS" \
    > "$OUT/extract_shards/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 100 "$OUT"/extract_shards/*.log || true
  exit 61
fi

BANK=$OUT/paired_vgg_r0.npz
python analysis/gan_tuning/paired_decoder_pullback_benchmark.py merge \
  --shard-dir "$OUT/extract_shards" --expected-shards "$SHARDS" \
  --output "$BANK" > "$OUT/merge.log" 2>&1

names=(standard_conditioned standard_z_only standard_shuffled highbandwidth_conditioned)
arms=(conditioned z_only shuffled conditioned)
architectures=(standard96 standard96 standard96 highbandwidth384)
pids=()
for gpu in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$gpu" python \
    analysis/gan_tuning/paired_decoder_pullback_benchmark.py fit \
    --bank "$BANK" --output "$OUT/results/${names[$gpu]}.json" \
    --arm "${arms[$gpu]}" --architecture "${architectures[$gpu]}" \
    --updates "$UPDATES" --batch-size 4 --eval-batch-size 4 --seeds 3 \
    --checkpoints "0,1,3,10,30,100,300,600,$UPDATES" \
    > "$OUT/results/${names[$gpu]}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 100 "$OUT"/results/*.log || true
  exit 62
fi

python - <<PY | tee "$OUT/SUMMARY.txt"
import json
from pathlib import Path
out = Path("$OUT")
for name in (
    "standard_conditioned", "standard_z_only", "standard_shuffled",
    "highbandwidth_conditioned",
):
    result = json.load(open(out / "results" / f"{name}.json"))
    metrics = [row["final_test"] for row in result["results"]]
    curves = [row["curve_test"] for row in result["results"]]
    print("ARM", name, "params", result["parameters"],
          "presentations", result["target_presentations"])
    print("test worst_q1", min(row["cosine"]["q1"] for row in metrics),
          "worst_median", min(row["cosine"]["median"] for row in metrics),
          "relative_mse_median_worst", max(row["relative_mse"]["median"] for row in metrics),
          "rms_ratio_medians", [row["rms_ratio"]["median"] for row in metrics])
    print("curve_worst_q1", {
        step: min(curve[step]["cosine"]["q1"] for curve in curves)
        for step in curves[0]
    })
PY

echo "[PAIRED-DECODER-PULLBACK] complete holder=$ALLOC output=$OUT"
