#!/bin/bash
# Compare detached RGB max/min conditioning variants on one holder node.
set -euo pipefail

: "${HOLDER:?set HOLDER to the active allocation}"
: "${NODE_INDEX:?set NODE_INDEX to select a holder node}"
: "${TARGET:?set TARGET to an extracted target bank}"
: "${OUT:?set OUT to an isolated result directory}"
: "${TEACHER_MODE:?set TEACHER_MODE to online or converged}"

cd /scratch/u6ex/as1748.u6ex/ARRWM
node_expr=$(squeue -j "$HOLDER" -h -o %N)
mapfile -t nodes < <(scontrol show hostnames "$node_expr")
node=${nodes[$NODE_INDEX]:-${nodes[0]}}
mkdir -p "$OUT"

if [ "$TEACHER_MODE" = converged ]; then
  checkpoint_args=(--checkpoints 1,3,6,12,24,48,96,192,384,768)
else
  checkpoint_args=(--substeps-per-step 24)
fi

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
checkpoint_args=('"$(printf '%q ' "${checkpoint_args[@]}")"')

# Keep the matrix compact and decision-facing: the strongest spatial MSE
# baseline, direction-only loss, global/temporal context, faster fitting, and
# a capacity check. Every arm receives the exact same detached RGB max/min
# guide aligned to the latent grid.
CANDIDATES=(
  "rgb_spatial_mse_lr2e4 spatial mse 2e-4 96 6 1e-3 24"
  "rgb_spatial_mse_lr1e3 spatial mse 1e-3 96 6 1e-3 24"
  "rgb_spatial_cos_lr1e3 spatial cosine 1e-3 96 6 1e-3 24"
  "rgb_temporalglobal_mse_lr1e3 temporal_global mse 1e-3 96 6 1e-3 24"
  "rgb_temporalglobal_cos_lr1e3 temporal_global cosine 1e-3 96 6 1e-3 24"
  "rgb_temporalglobal_cos_fit12_lr1e3 temporal_global cosine 1e-3 96 6 1e-3 12"
  "rgb_temporalglobal_cos_w128b8_lr1e3 temporal_global cosine 1e-3 128 8 1e-3 24"
)

pids=()
gpu=0
flush_wave() {
  local rc=0 p
  for p in "${pids[@]}"; do wait "$p" || rc=1; done
  pids=()
  test "$rc" -eq 0
}
for spec in "${CANDIDATES[@]}"; do
  read -r name arch loss lr width blocks head_init substeps <<< "$spec"
  extra=("${checkpoint_args[@]}")
  if [ "$TEACHER_MODE" = online ]; then
    extra=(--substeps-per-step "$substeps")
  fi
  CUDA_VISIBLE_DEVICES="$gpu" python \
    analysis/gan_tuning/surrogate_feature_field_benchmark.py fit-surrogate \
    --targets "$TARGET" --output "$OUT/${name}.json" \
    --condition rgbmaxmin --teacher-mode "$TEACHER_MODE" \
    --architecture "$arch" --loss-mode "$loss" --lr "$lr" \
    --width "$width" --blocks "$blocks" --head-init-std "$head_init" \
    --temporal-blocks 2 --seeds 3 "${extra[@]}" \
    > "$OUT/${name}.log" 2>&1 &
  pids+=("$!")
  gpu=$((gpu + 1))
  if [ "$gpu" -eq 4 ]; then flush_wave; gpu=0; fi
done
if [ "${#pids[@]}" -gt 0 ]; then flush_wave; fi

if [ "$TEACHER_MODE" = online ]; then
  python analysis/gan_tuning/summarize_surrogate_gate_screen.py \
    --input "$OUT" --output "$OUT/SUMMARY.md" \
    --expected-condition rgbmaxmin
else
  python analysis/gan_tuning/summarize_surrogate_frozen_screen.py \
    --input "$OUT" --output "$OUT/SUMMARY.md"
fi
'
