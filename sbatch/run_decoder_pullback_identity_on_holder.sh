#!/bin/bash
# Identity/preflight for the stationary WAN decoder-pullback benchmark.
# Runs inside an existing durable holder; never submits or cancels a job.
set -euo pipefail

: "${ALLOC:?set ALLOC to a running holder}"
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
DATA=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination
FIELD=$DATA/surrogate_field_2708
CAPTURE=$DATA/surrogate_capture_2708
OUT=${OUT:-$FIELD/decoder_pullback_identity_h${ALLOC}}
ROTATION=${ROTATION:-0}
GPU=${GPU:-0}

cd "$ROOT"
mkdir -p "$OUT"
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
export PYTHONPATH=. OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

python -m py_compile \
  model/decoder_pullback_surrogate.py \
  analysis/gan_tuning/decoder_pullback_benchmark.py \
  testing/test_decoder_pullback_surrogate.py
python -m pytest -q testing/test_decoder_pullback_surrogate.py \
  > "$OUT/model_tests.log" 2>&1

CUDA_VISIBLE_DEVICES="$GPU" python \
  analysis/gan_tuning/decoder_pullback_benchmark.py extract \
  --capture-dir "$CAPTURE" \
  --head "$FIELD/head_curves/vgg_r${ROTATION}_lr1e3.pt" \
  --rotation "$ROTATION" --crop-index 0 \
  --max-z-per-role 1 \
  --output "$OUT/vgg_r${ROTATION}_identity_smoke.npz" \
  > "$OUT/vgg_r${ROTATION}_identity_smoke.log" 2>&1

python - <<PY
import json
import numpy as np
from pathlib import Path
p = Path("$OUT/vgg_r${ROTATION}_identity_smoke.npz")
with np.load(p, allow_pickle=False) as bank:
    meta = json.loads(str(bank["metadata"].item()))
    assert bank["latent"].shape == (2, 3, 16, 24, 32)
    assert bank["pixel_cotangent"].shape == (8, 12, 3, 192, 256)
    assert bank["pullback"].shape == (2, 8, 3, 16, 24, 32)
identity = meta["identity"]
assert identity["cosine"] >= 0.99999, identity
assert identity["scale_fitted_relative_error"] <= 1e-3, identity
print(json.dumps({
    "identity": identity,
    "seconds": meta["seconds"],
    "peak_allocated_gib": meta["peak_allocated_gib"],
    "geometry": meta["geometry"],
}, indent=2))
PY

echo "[DECODER-PULLBACK] identity PASS output=$OUT"
