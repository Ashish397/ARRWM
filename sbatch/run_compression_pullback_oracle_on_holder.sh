#!/bin/bash
# Exact block-projection decoder-VJP oracle on four GPUs of a live holder.
set -euo pipefail
: "${ALLOC:?set ALLOC to the live holder job id}"

ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
DATA=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination
FIELD=$DATA/surrogate_field_2708
BANK=${BANK:-$FIELD/decoder_pullback_zscale_r0_h6156714/vgg_r0_live_pullback_z3.npz}
OUT=${OUT:-$FIELD/compression_pullback_oracle_r0_h${ALLOC}}
SHARDS=${SHARDS:-4}

if [ "$SHARDS" -ne 4 ]; then
  echo "This reviewed holder command expects exactly four local GPU shards" >&2
  exit 64
fi
if [ ! -f "$BANK" ]; then
  echo "Missing exact pullback bank: $BANK" >&2
  exit 66
fi

cd "$ROOT"
mkdir -p "$OUT/shards"
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2

python -m py_compile \
  analysis/gan_tuning/compression_pullback_oracle.py \
  testing/test_compression_pullback_oracle.py
python -m pytest -q testing/test_compression_pullback_oracle.py \
  > "$OUT/tests.log" 2>&1

pids=()
for shard in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES="$shard" python \
    analysis/gan_tuning/compression_pullback_oracle.py run-shard \
    --bank "$BANK" \
    --output "$OUT/shards/shard_${shard}.json" \
    --z-role test --v-role test \
    --shard-index "$shard" --num-shards "$SHARDS" \
    > "$OUT/shards/shard_${shard}.log" 2>&1 &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 100 "$OUT"/shards/*.log || true
  exit 61
fi

python analysis/gan_tuning/compression_pullback_oracle.py merge \
  --shard-dir "$OUT/shards" --expected-shards "$SHARDS" \
  --output "$OUT/ORACLE.json" > "$OUT/merge.log" 2>&1

python - <<PY | tee "$OUT/SUMMARY.txt"
import json
result = json.load(open("$OUT/ORACLE.json"))
print("verdict", result["t4_s8_verdict"])
print("pairs", result["pair_count"])
print("factor pixel_rms_q1 pixel_rms_med pullback_cos_q1 pullback_cos_med relmse_med rmsratio_med")
for name, row in result["summaries"].items():
    print(name,
          row["pixel_projected_rms_ratio"]["q1"],
          row["pixel_projected_rms_ratio"]["median"],
          row["pullback_cosine"]["q1"],
          row["pullback_cosine"]["median"],
          row["pullback_relative_mse"]["median"],
          row["pullback_rms_ratio"]["median"])
print("scope", result["interpretation_scope"])
PY

echo "[COMPRESSION-PULLBACK-ORACLE] complete holder=$ALLOC output=$OUT"
