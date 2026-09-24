#!/bin/bash
# Build the exact ARRWM nine-latent/three-action seed bundles on one holder GPU.
set -euo pipefail
ALLOC=${1:?usage: panel32_seed_bundles_on_holder.sh ALLOC PANEL_MANIFEST SOURCE_PROVENANCE [NODE]}
PANEL_MANIFEST=${2:?usage: panel32_seed_bundles_on_holder.sh ALLOC PANEL_MANIFEST SOURCE_PROVENANCE [NODE]}
SOURCE_PROVENANCE=${3:?usage: panel32_seed_bundles_on_holder.sh ALLOC PANEL_MANIFEST SOURCE_PROVENANCE [NODE]}
NODE=${4:-$(squeue -h -j "$ALLOC" -o '%N' | head -1)}
test "$(squeue -h -j "$ALLOC" -o '%T' | head -1)" = RUNNING
R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
PY=${ARRWM_PYTHON:-$R/miniforge3/envs/arrwm/bin/python}
OUT=${PANEL32_BUNDLE_ROOT:-$R/panel32_stage/ours_seed_bundles}
LOG=${PANEL32_BUNDLE_LOG:-$R/panel32_stage/logs/ours_seed_bundles.log}
mkdir -p "$OUT" "$(dirname "$LOG")"
exec srun --overlap --jobid="$ALLOC" --nodes=1 --nodelist="$NODE" \
  --ntasks=1 --cpus-per-task=16 --gpus=1 --gpu-bind=single:1 \
  bash -lc "export PYTHONPATH='$R/torch_home/hub/facebookresearch_co-tracker_main:$A'; export PATH='$R/miniforge3/envs/arrwm/bin':\"\$PATH\"; export TORCH_HOME='$R/torch_home'; export HF_HOME='$R/hf_cache'; '$PY' '$A/grids/eval/panel32_seed_bundles.py' build --panel-manifest '$PANEL_MANIFEST' --source-provenance '$SOURCE_PROVENANCE' --output '$OUT' --wan-model-root '$R/frodobots' --pca-checkpoint '$A/code_release/preprocessing/checkpoints/pca_basis.pt' --cotracker-repo '$R/torch_home/hub/facebookresearch_co-tracker_main' 2>&1 | tee '$LOG'"
