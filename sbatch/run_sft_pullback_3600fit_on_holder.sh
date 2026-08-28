#!/bin/bash
# Equal-3,600-presentation continuation screen using the completed SFT banks.
set -euo pipefail
: "${ALLOC:?set ALLOC to the live holder job id}"

ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
DATA=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination
FIELD=$DATA/surrogate_field_2708
OUT=${OUT:-$FIELD/sft_pullback_r0_h${ALLOC}}
PAIRED=${PAIRED:-$FIELD/paired_decoder_pullback_r0_h6158256/paired_vgg_r0.npz}
STATE_BANK=${STATE_BANK:-$OUT/detached_wan_state.npz}
RESULTS=$OUT/results_3600presentations

if [ ! -f "$PAIRED" ] || [ ! -f "$STATE_BANK" ]; then
  echo "Missing paired or decoder-state bank" >&2
  exit 66
fi
cd "$ROOT"
mkdir -p "$RESULTS"
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

names=(sft sft_state sft_shuffled)
pids=()
for gpu in 0 1 2; do
  extra=()
  if [ "${names[$gpu]}" = "sft_state" ]; then
    extra+=(--state-bank "$STATE_BANK")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python \
    analysis/gan_tuning/sft_pullback_benchmark.py fit \
    --paired-bank "$PAIRED" "${extra[@]}" \
    --output "$RESULTS/${names[$gpu]}.json" \
    --arm "${names[$gpu]}" --updates 1800 \
    --batch-size 2 --eval-batch-size 2 --seeds 3 \
    --checkpoints "0,3,10,30,100,300,600,900,1200,1500,1800" \
    > "$RESULTS/${names[$gpu]}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 120 "$RESULTS"/*.log || true
  exit 62
fi

python - <<PY | tee "$OUT/SUMMARY_3600P.txt"
import json
from pathlib import Path
results = Path("$RESULTS")
for name in ("sft", "sft_state", "sft_shuffled"):
    result = json.load(open(results / f"{name}.json"))
    rows = result["results"]
    final_test = [row["final_test"] for row in rows]
    final_train = [row["final_train"] for row in rows]
    curves = [row["curve_test"] for row in rows]
    curve_q1 = {
        step: min(curve[step]["cosine"]["q1"] for curve in curves)
        for step in curves[0]
    }
    curve_median = {
        step: min(curve[step]["cosine"]["median"] for curve in curves)
        for step in curves[0]
    }
    best_step, best_q1 = max(curve_q1.items(), key=lambda item: item[1])
    print("ARM", name, "params", result["parameters"],
          "presentations", result["target_presentations"])
    print("best_robust_q1", best_q1, "best_step", best_step,
          "best_step_worst_median", curve_median[best_step],
          "final_test_q1", min(row["cosine"]["q1"] for row in final_test),
          "final_test_median", min(row["cosine"]["median"] for row in final_test),
          "final_train_q1", min(row["cosine"]["q1"] for row in final_train),
          "final_train_median", min(row["cosine"]["median"] for row in final_train),
          "final_relative_mse_median", max(row["relative_mse"]["median"] for row in final_test))
    print("curve_worst_q1", curve_q1)
PY

echo "[SFT-PULLBACK] 3,600-presentation screen complete holder=$ALLOC output=$OUT"
