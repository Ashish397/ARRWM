#!/bin/bash
# Paired real+fake versus fake-only field ablation inside one holder node.
# This launcher never submits or cancels an allocation.
set -euo pipefail

HOLDER=${HOLDER:-${ALLOC:-}}
: "${HOLDER:?set HOLDER or ALLOC to a running holder}"
: "${SELECTED_CANDIDATE:?name the robust cross-rotation gate winner}"
: "${DOMAIN_ARCHITECTURE:?set spatial|temporal|global|temporal_global from winner}"
: "${DOMAIN_LOSS_MODE:?set mse|cosine from winner}"
: "${DOMAIN_STUDENT_LR:?set student LR from winner}"
: "${DOMAIN_HEAD_INIT:?set direct head initialization from winner}"
: "${DOMAIN_WIDTH:?set direct predictor width from winner}"
: "${DOMAIN_BLOCKS:?set direct predictor block count from winner}"
: "${DOMAIN_SUBSTEPS:?set fits per teacher version from winner}"
case "$DOMAIN_ARCHITECTURE" in
  spatial|temporal|global|temporal_global) ;;
  *) echo "invalid DOMAIN_ARCHITECTURE=$DOMAIN_ARCHITECTURE" >&2; exit 2 ;;
esac
case "$DOMAIN_LOSS_MODE" in
  mse|cosine) ;;
  *) echo "invalid DOMAIN_LOSS_MODE=$DOMAIN_LOSS_MODE" >&2; exit 2 ;;
esac
for value in "$DOMAIN_WIDTH" "$DOMAIN_BLOCKS" "$DOMAIN_SUBSTEPS"; do
  [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
    echo "width, blocks and substeps must be positive integers" >&2; exit 2;
  }
done
DOMAIN_FAKE_SUBSTEPS=$((2 * DOMAIN_SUBSTEPS))

ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
DATA=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination
CAPTURE=${CAPTURE:-$DATA/surrogate_capture_2708}
FIELD=$DATA/surrogate_field_2708
OUT=${OUT:-$FIELD/domain_balance_2708}
NODE_INDEX=${NODE_INDEX:-0}
SCRIPT=analysis/gan_tuning/surrogate_domain_balance_benchmark.py

cd "$ROOT"
NODE_EXPR=$(squeue -j "$HOLDER" -h -o %N)
mapfile -t NODES < <(scontrol show hostnames "$NODE_EXPR")
test "${#NODES[@]}" -gt 0
if [ "$NODE_INDEX" -ge "${#NODES[@]}" ]; then NODE_INDEX=0; fi
NODE=${NODES[$NODE_INDEX]}
mkdir -p "$OUT/targets" "$OUT/results" logs
LOG=logs/surrogate_domain_balance_h${HOLDER}_$(date +%Y%m%d_%H%M%S).log

# A prior holder command may have returned just before Slurm reaped its step.
# Wait briefly and fail closed rather than colliding with any non-batch child.
LIVE=1
for _attempt in $(seq 1 30); do
  LIVE=$(squeue -j "$HOLDER" -h -s -o %i 2>/dev/null \
    | grep -vE '\.(batch|extern)$' | wc -l) || true
  [ "${LIVE:-0}" -eq 0 ] && break
  sleep 2
done
[ "${LIVE:-0}" -eq 0 ] || {
  echo "[SURROGATE-DOMAIN] FATAL: holder has $LIVE live child step(s)" >&2
  exit 9
}

echo "[SURROGATE-DOMAIN] holder=$HOLDER node=$NODE winner=$SELECTED_CANDIDATE" | tee "$LOG"
srun --jobid="$HOLDER" --overlap --nodelist="$NODE" --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --gpu-bind=none bash -lc '
set -euo pipefail
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=. OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
CAPTURE='"$CAPTURE"'
FIELD='"$FIELD"'
OUT='"$OUT"'
SCRIPT='"$SCRIPT"'

pids=()
for rotation in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$rotation" python "$SCRIPT" extract-paired \
    --capture-dir "$CAPTURE" \
    --head "$FIELD/head_curves/vgg_r${rotation}_lr1e3.pt" \
    --source vgg --rotation "$rotation" --crop-index 0 \
    --output "$OUT/targets/vgg_r${rotation}_paired_lr1e3.npz" \
    > "$OUT/targets/vgg_r${rotation}.log" 2>&1 &
  pids+=("$!")
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then tail -80 "$OUT"/targets/*.log || true; exit 31; fi

pids=()
for rotation in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$rotation" python "$SCRIPT" fit-comparison \
    --targets "$OUT/targets/vgg_r${rotation}_paired_lr1e3.npz" \
    --output "$OUT/results/vgg_r${rotation}.json" \
    --architecture '"$DOMAIN_ARCHITECTURE"' \
    --loss-mode '"$DOMAIN_LOSS_MODE"' \
    --lr '"$DOMAIN_STUDENT_LR"' \
    --head-init-std '"$DOMAIN_HEAD_INIT"' \
    --width '"$DOMAIN_WIDTH"' --blocks '"$DOMAIN_BLOCKS"' \
    --substeps-per-step '"$DOMAIN_SUBSTEPS"' \
    --fake-compute-substeps '"$DOMAIN_FAKE_SUBSTEPS"' --seeds 3 \
    > "$OUT/results/vgg_r${rotation}.log" 2>&1 &
  pids+=("$!")
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then tail -80 "$OUT"/results/*.log || true; exit 32; fi

python analysis/gan_tuning/summarize_surrogate_domain_balance.py \
  --inputs "$OUT/results/vgg_r0.json" "$OUT/results/vgg_r1.json" \
           "$OUT/results/vgg_r2.json" \
  --selected-candidate '"$SELECTED_CANDIDATE"' \
  --output "$OUT/SUMMARY.md"
' >> "$LOG" 2>&1

echo "[SURROGATE-DOMAIN] complete summary=$OUT/SUMMARY.md" | tee -a "$LOG"
