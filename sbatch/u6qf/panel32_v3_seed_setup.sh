#!/bin/bash
# Build and validate the corrected panel's ARRWM seed bundles inside a holder.
set -euo pipefail

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=${PANEL32_STAGE:-$R/panel32_v3_stage}
PY=${ARRWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python}
MANIFEST=$A/grids/eval/panel32_locked_v3.json
PROVENANCE=$P/sources/panel32_source_provenance.json
OUTPUT=$P/ours_seed_bundles
LOG=$P/logs/seed_setup

mkdir -p "$LOG"
export PATH="$R/miniforge3/envs/arrwm/bin:$PATH"
export PYTHONPATH="$R/torch_home/hub/facebookresearch_co-tracker_main:$A${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME=${HF_HOME:-$R/frodobots/hf_cache}
export TORCH_HOME=${TORCH_HOME:-$R/torch_home}
export CUDA_VISIBLE_DEVICES=0

"$PY" "$A/grids/eval/panel32_seed_bundles.py" build \
  --panel-manifest "$MANIFEST" \
  --source-provenance "$PROVENANCE" \
  --output "$OUTPUT" \
  --wan-model-root "$R/frodobots" \
  --pca-checkpoint "$A/code_release/preprocessing/checkpoints/pca_basis.pt" \
  --cotracker-repo "$R/torch_home/hub/facebookresearch_co-tracker_main" \
  >"$LOG/build.log" 2>&1

"$PY" "$A/grids/eval/panel32_seed_bundles.py" validate \
  --panel-manifest "$MANIFEST" \
  --source-provenance "$PROVENANCE" \
  --output "$OUTPUT" \
  >"$LOG/validate.log" 2>&1

touch "$LOG/COMPLETE"
echo "PANEL32_V3_SEED_SETUP_COMPLETE $(date -Is)"
