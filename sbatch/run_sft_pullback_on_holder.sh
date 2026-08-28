#!/bin/bash
# Three-arm naturally paired multiscale SFT screen on a live holder.
set -euo pipefail
: "${ALLOC:?set ALLOC to the live holder job id}"

ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
DATA=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination
FIELD=$DATA/surrogate_field_2708
PAIRED=${PAIRED:-$FIELD/paired_decoder_pullback_r0_h6158256/paired_vgg_r0.npz}
OUT=${OUT:-$FIELD/sft_pullback_r0_h${ALLOC}}
UPDATES=${UPDATES:-900}
SHARDS=4

cd "$ROOT"
mkdir -p "$OUT/state_shards" "$OUT/results"
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

python -m py_compile \
  model/sft_pullback_surrogate.py \
  analysis/gan_tuning/sft_pullback_benchmark.py
python -m pytest -q testing/test_sft_pullback_surrogate.py \
  > "$OUT/sft_tests.log" 2>&1

STATE_BANK=$OUT/detached_wan_state.npz
if [ ! -f "$STATE_BANK" ]; then
  pids=()
  for shard in 0 1 2 3; do
    CUDA_VISIBLE_DEVICES="$shard" python \
      analysis/gan_tuning/sft_pullback_benchmark.py extract-state-shard \
      --paired-bank "$PAIRED" \
      --output "$OUT/state_shards/state_shard_${shard}.npz" \
      --shard-index "$shard" --num-shards "$SHARDS" \
      > "$OUT/state_shards/state_shard_${shard}.log" 2>&1 &
    pids+=("$!")
  done
  rc=0
  for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
  if [ "$rc" -ne 0 ]; then
    tail -n 100 "$OUT"/state_shards/*.log || true
    exit 61
  fi
  python analysis/gan_tuning/sft_pullback_benchmark.py merge-state \
    --shard-dir "$OUT/state_shards" --expected-shards "$SHARDS" \
    --output "$STATE_BANK" > "$OUT/state_merge.log" 2>&1
fi

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
    --output "$OUT/results/${names[$gpu]}.json" \
    --arm "${names[$gpu]}" --updates "$UPDATES" \
    --batch-size 2 --eval-batch-size 2 --seeds 3 \
    --checkpoints "0,3,10,30,100,300,600,$UPDATES" \
    > "$OUT/results/${names[$gpu]}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 120 "$OUT"/results/*.log || true
  exit 62
fi

python - <<PY | tee "$OUT/SUMMARY.txt"
import json
from pathlib import Path
out = Path("$OUT")
for name in ("sft", "sft_state", "sft_shuffled"):
    result = json.load(open(out / "results" / f"{name}.json"))
    rows = result["results"]
    final_test = [row["final_test"] for row in rows]
    final_train = [row["final_train"] for row in rows]
    curves = [row["curve_test"] for row in rows]
    curve_q1 = {
        step: min(curve[step]["cosine"]["q1"] for curve in curves)
        for step in curves[0]
    }
    best_step, best_q1 = max(curve_q1.items(), key=lambda item: item[1])
    print("ARM", name, "params", result["parameters"],
          "presentations", result["target_presentations"])
    print("best_robust_q1", best_q1, "best_step", best_step,
          "final_test_q1", min(row["cosine"]["q1"] for row in final_test),
          "final_test_median", min(row["cosine"]["median"] for row in final_test),
          "final_train_q1", min(row["cosine"]["q1"] for row in final_train),
          "final_train_median", min(row["cosine"]["median"] for row in final_train))
    print("curve_worst_q1", curve_q1)
PY

echo "[SFT-PULLBACK] three-arm screen complete holder=$ALLOC output=$OUT"
