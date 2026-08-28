#!/bin/bash
# Deterministic stop-at-selected-peak train/test audit for SFT and shuffled SFT.
set -euo pipefail
: "${ALLOC:?set ALLOC}"
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
FIELD=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708
OUT=${OUT:-$FIELD/sft_pullback_r0_h${ALLOC}}
PAIRED=$FIELD/paired_decoder_pullback_r0_h6158256/paired_vgg_r0.npz
RESULTS=$OUT/peak_refit
cd "$ROOT"
mkdir -p "$RESULTS"
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

pids=()
for spec in "0:sft" "1:sft_shuffled"; do
  gpu=${spec%%:*}; arm=${spec#*:}
  CUDA_VISIBLE_DEVICES="$gpu" python \
    analysis/gan_tuning/sft_pullback_benchmark.py fit \
    --paired-bank "$PAIRED" --output "$RESULTS/${arm}.json" \
    --arm "$arm" --updates 600 --batch-size 2 --eval-batch-size 2 \
    --seeds 3 --checkpoints "0,600" \
    > "$RESULTS/${arm}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 100 "$RESULTS"/*.log || true
  exit 62
fi
python - <<PY | tee "$OUT/PEAK_TRAIN_AUDIT.txt"
import json
from pathlib import Path
root = Path("$RESULTS")
for arm in ("sft", "sft_shuffled"):
    result = json.load(open(root / f"{arm}.json"))
    rows = result["results"]
    print(arm,
          "test_q1", min(row["final_test"]["cosine"]["q1"] for row in rows),
          "test_median", min(row["final_test"]["cosine"]["median"] for row in rows),
          "train_q1", min(row["final_train"]["cosine"]["q1"] for row in rows),
          "train_median", min(row["final_train"]["cosine"]["median"] for row in rows))
PY
