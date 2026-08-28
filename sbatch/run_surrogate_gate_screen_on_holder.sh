#!/bin/bash
# Honest ride-disjoint screen for the direct-gradient surrogate candidates.
# Uses one node inside an existing holder; it never submits or cancels jobs.
set -euo pipefail

HOLDER=${HOLDER:-${ALLOC:-}}
: "${HOLDER:?set HOLDER or ALLOC to a running holder}"
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
BASE=${BASE:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708}
TARGET=${TARGET:-$BASE/targets_fasthead/vgg_r0_headlr1e3.npz}
OUT=${OUT:-$BASE/results_gate_v2}
NODE_INDEX=${NODE_INDEX:-0}
SCREEN_TAG=${SCREEN_TAG:-$(basename "$OUT")}
SCREEN_TAG=${SCREEN_TAG//[^A-Za-z0-9_.-]/_}

cd "$ROOT"
NODE_EXPR=$(squeue -j "$HOLDER" -h -o %N)
mapfile -t NODES < <(scontrol show hostnames "$NODE_EXPR")
test "${#NODES[@]}" -gt 0
if [ "$NODE_INDEX" -ge "${#NODES[@]}" ]; then NODE_INDEX=0; fi
NODE=${NODES[$NODE_INDEX]}
mkdir -p "$OUT" logs
LOG=logs/surrogate_gate_v2_h${HOLDER}_${SCREEN_TAG}_$(date +%Y%m%d_%H%M%S).log

echo "[SURROGATE-GATE-V2] holder=$HOLDER node=$NODE target=$TARGET" | tee "$LOG"
srun --jobid="$HOLDER" --overlap --nodelist="$NODE" --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --gpu-bind=none bash -lc '
set -euo pipefail
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=. OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
TARGET='"$TARGET"'
OUT='"$OUT"'
SCRIPT=analysis/gan_tuning/surrogate_feature_field_benchmark.py

# name architecture loss lr width blocks head_init substeps
CANDIDATES=(
  "spatial_mse_lr2e4 spatial mse 2e-4 96 6 1e-3 24"
  "spatial_mse_lr1e3 spatial mse 1e-3 96 6 1e-3 24"
  "spatial_cos_lr1e3 spatial cosine 1e-3 96 6 1e-3 24"
  "spatial_cos_head1e2_lr1e3 spatial cosine 1e-3 96 6 1e-2 24"
  "temporal_cos_lr1e3 temporal cosine 1e-3 96 6 1e-3 24"
  "global_cos_lr1e3 global cosine 1e-3 96 6 1e-3 24"
  "temporalglobal_mse_lr1e3 temporal_global mse 1e-3 96 6 1e-3 24"
  "temporalglobal_cos_lr1e3 temporal_global cosine 1e-3 96 6 1e-3 24"
  "temporalglobal_cos_lr2e3 temporal_global cosine 2e-3 96 6 1e-3 24"
  "temporalglobal_cos_head1e2_lr1e3 temporal_global cosine 1e-3 96 6 1e-2 24"
  "temporalglobal_cos_head2e2_lr1e3 temporal_global cosine 1e-3 96 6 2e-2 24"
  "temporalglobal_cos_fit12_lr1e3 temporal_global cosine 1e-3 96 6 1e-3 12"
  "temporalglobal_cos_fit48_lr1e3 temporal_global cosine 1e-3 96 6 1e-3 48"
  "temporalglobal_cos_w128b8_lr1e3 temporal_global cosine 1e-3 128 8 1e-3 24"
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
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" fit-surrogate \
    --targets "$TARGET" --output "$OUT/${name}.json" \
    --condition latent --teacher-mode online --architecture "$arch" \
    --loss-mode "$loss" --lr "$lr" --width "$width" --blocks "$blocks" \
    --head-init-std "$head_init" --temporal-blocks 2 \
    --substeps-per-step "$substeps" --seeds 3 \
    > "$OUT/${name}.log" 2>&1 &
  pids+=("$!")
  gpu=$((gpu + 1))
  if [ "$gpu" -eq 4 ]; then flush_wave; gpu=0; fi
done
if [ "${#pids[@]}" -gt 0 ]; then flush_wave; fi

python analysis/gan_tuning/summarize_surrogate_gate_screen.py \
  --input "$OUT" --output "$OUT/SUMMARY.md"
' >> "$LOG" 2>&1

echo "[SURROGATE-GATE-V2] complete summary=$OUT/SUMMARY.md" | tee -a "$LOG"
