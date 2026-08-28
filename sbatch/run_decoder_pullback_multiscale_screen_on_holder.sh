#!/bin/bash
# Matched v1 versus multiscale-z-gated v2 pullback fit on an existing bank.
set -euo pipefail
: "${ALLOC:?set ALLOC}"
: "${BANK:?set BANK}"
: "${OUT:?set OUT}"

cd /scratch/u6ex/as1748.u6ex/ARRWM
mkdir -p "$OUT"
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export PYTHONPATH=. OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
python -m py_compile \
  model/decoder_pullback_surrogate.py \
  analysis/gan_tuning/decoder_pullback_benchmark.py
python -m pytest -q testing/test_decoder_pullback_surrogate.py \
  > "$OUT/model_tests.log" 2>&1

SPECS=(
  "v1_conditioned conditioned 0 off"
  "v2_conditioned conditioned 1 on"
  "v2_z_only z_only 2 on"
  "v2_shuffled shuffled 3 on"
)
pids=()
for spec in "${SPECS[@]}"; do
  read -r name arm gpu gating <<< "$spec"
  extra=()
  if [ "$gating" = on ]; then extra+=(--multiscale-z-gating); fi
  CUDA_VISIBLE_DEVICES="$gpu" python \
    analysis/gan_tuning/decoder_pullback_benchmark.py fit \
    --bank "$BANK" --output "$OUT/${name}.json" --arm "$arm" \
    --updates 600 --checkpoints 0,1,3,10,30,100,300,600 --seeds 3 \
    --batch-size 2 --eval-batch-size 2 --width 96 --blocks 4 \
    "${extra[@]}" > "$OUT/${name}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 120 "$OUT"/*.log || true
  exit 71
fi

python - <<PY | tee "$OUT/SUMMARY.txt"
import json
from pathlib import Path
out = Path("$OUT")
for name in ("v1_conditioned", "v2_conditioned", "v2_z_only", "v2_shuffled"):
    d = json.load(open(out / f"{name}.json"))
    print("ARM", name, "params", d["parameters"],
          "presentations", d["target_presentations"],
          "multiscale", d["multiscale_z_gating"])
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
    print("max_structural_error", max(
        max(r["structural"].values()) for r in d["results"]
    ))
PY
echo "[DECODER-PULLBACK] multiscale screen complete output=$OUT"
