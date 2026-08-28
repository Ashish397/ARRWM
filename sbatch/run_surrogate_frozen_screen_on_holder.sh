#!/bin/bash
# Frozen-teacher representational/optimization ceiling on one holder node.
set -euo pipefail

HOLDER=${HOLDER:-${ALLOC:-}}
: "${HOLDER:?set HOLDER or ALLOC to a running holder}"
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
BASE=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708
TARGET=${TARGET:?set exact target bank}
OUT=${OUT:?set isolated output directory}
NODE_INDEX=${NODE_INDEX:-0}
SCREEN_TAG=${SCREEN_TAG:-frozen}

cd "$ROOT"
node_expr=$(squeue -j "$HOLDER" -h -o %N)
mapfile -t nodes < <(scontrol show hostnames "$node_expr")
node=${nodes[$NODE_INDEX]:-${nodes[0]}}
mkdir -p "$OUT" logs
log=logs/surrogate_frozen_h${HOLDER}_${SCREEN_TAG}_$(date +%Y%m%d_%H%M%S).log

srun --jobid="$HOLDER" --overlap --nodelist="$node" --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --gpu-bind=none bash -lc '
set -euo pipefail
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=. OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
TARGET='"$TARGET"'; OUT='"$OUT"'
SCRIPT=analysis/gan_tuning/surrogate_feature_field_benchmark.py
# name architecture loss lr width blocks head_init
CANDIDATES=(
  "spatial_mse_lr1e3 spatial mse 1e-3 96 6 1e-3"
  "spatial_cos_lr1e3 spatial cosine 1e-3 96 6 1e-3"
  "global_mse_lr1e3 global mse 1e-3 96 6 1e-3"
  "temporal_mse_lr1e3 temporal mse 1e-3 96 6 1e-3"
  "temporalglobal_mse_lr1e3 temporal_global mse 1e-3 96 6 1e-3"
  "temporalglobal_cos_lr1e3 temporal_global cosine 1e-3 96 6 1e-3"
  "temporalglobal_mse_lr2e4 temporal_global mse 2e-4 96 6 1e-3"
  "temporalglobal_mse_w128b8_lr1e3 temporal_global mse 1e-3 128 8 1e-3"
)
pids=(); gpu=0
flush_wave() {
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  pids=(); test "$rc" -eq 0
}
for spec in "${CANDIDATES[@]}"; do
  read -r name arch loss lr width blocks head_init <<< "$spec"
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" fit-surrogate \
    --targets "$TARGET" --output "$OUT/${name}.json" \
    --condition latent --teacher-mode converged --architecture "$arch" \
    --loss-mode "$loss" --lr "$lr" --width "$width" --blocks "$blocks" \
    --head-init-std "$head_init" --temporal-blocks 2 --seeds 3 \
    --checkpoints 1,3,6,12,24,48,96,192,384,768 \
    > "$OUT/${name}.log" 2>&1 &
  pids+=("$!"); gpu=$((gpu + 1))
  if [ "$gpu" -eq 4 ]; then flush_wave; gpu=0; fi
done
if [ "${#pids[@]}" -gt 0 ]; then flush_wave; fi
python analysis/gan_tuning/summarize_surrogate_frozen_screen.py \
  --input "$OUT" --output "$OUT/SUMMARY.md"
' > "$log" 2>&1
echo "[SURROGATE-FROZEN] complete holder=$HOLDER summary=$OUT/SUMMARY.md" | tee -a "$log"
