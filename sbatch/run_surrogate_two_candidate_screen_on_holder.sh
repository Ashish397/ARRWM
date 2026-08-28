#!/bin/bash
# Two-candidate matched screen for a named surrogate condition.
set -euo pipefail
: "${HOLDER:?set HOLDER}"
: "${NODE_INDEX:?set NODE_INDEX}"
: "${TARGET:?set TARGET}"
: "${OUT:?set OUT}"
: "${TEACHER_MODE:?set TEACHER_MODE}"
: "${CONDITION:?set CONDITION}"
: "${PREFIX:?set PREFIX}"

cd /scratch/u6ex/as1748.u6ex/ARRWM
node_expr=$(squeue -j "$HOLDER" -h -o %N)
mapfile -t nodes < <(scontrol show hostnames "$node_expr")
node=${nodes[$NODE_INDEX]:-${nodes[0]}}
mkdir -p "$OUT"
eval_crop=${EVAL_CROP_INDEX:-}

srun --jobid="$HOLDER" --overlap --nodelist="$node" \
  --nodes=1 --ntasks=1 --gpus-per-node=4 --gpu-bind=none bash -lc '
set -euo pipefail
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=. OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
TARGET='"$(printf %q "$TARGET")"'
OUT='"$(printf %q "$OUT")"'
TEACHER_MODE='"$(printf %q "$TEACHER_MODE")"'
CONDITION='"$(printf %q "$CONDITION")"'
PREFIX='"$(printf %q "$PREFIX")"'
EVAL_CROP='"$(printf %q "$eval_crop")"'

CANDIDATES=(
  "${PREFIX}_spatial_mse_lr1e3 spatial"
  "${PREFIX}_temporalglobal_mse_lr1e3 temporal_global"
)
pids=()
gpu=0
for spec in "${CANDIDATES[@]}"; do
  read -r name arch <<< "$spec"
  extra=()
  if [ -n "$EVAL_CROP" ]; then extra+=(--eval-crop-index "$EVAL_CROP"); fi
  if [ "$TEACHER_MODE" = online ]; then
    extra+=(--substeps-per-step 24)
  else
    extra+=(--checkpoints 1,3,6,12,24,48,96,192,384,768)
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python \
    analysis/gan_tuning/surrogate_feature_field_benchmark.py fit-surrogate \
    --targets "$TARGET" --output "$OUT/${name}.json" \
    --condition "$CONDITION" --teacher-mode "$TEACHER_MODE" \
    --architecture "$arch" --loss-mode mse --lr 1e-3 \
    --width 96 --blocks 6 --head-init-std 1e-3 --temporal-blocks 2 \
    --seeds 3 "${extra[@]}" > "$OUT/${name}.log" 2>&1 &
  pids+=("$!")
  gpu=$((gpu + 1))
done
rc=0
for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -n 80 "$OUT"/*.log || true
  exit 41
fi
if [ "$TEACHER_MODE" = online ]; then
  python analysis/gan_tuning/summarize_surrogate_gate_screen.py \
    --input "$OUT" --output "$OUT/SUMMARY.md" \
    --expected-condition "$CONDITION"
else
  python analysis/gan_tuning/summarize_surrogate_frozen_screen.py \
    --input "$OUT" --output "$OUT/SUMMARY.md"
fi
'

