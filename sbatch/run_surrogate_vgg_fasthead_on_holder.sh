#!/bin/bash
# Follow-up field test for the fastest VGG discriminator head (LR=1e-3).
# Runs on one otherwise-idle node inside an existing holder. Never submits or
# cancels the allocation and never touches the primary 2e-4 result files.
set -euo pipefail

HOLDER=${HOLDER:-${ALLOC:-}}
: "${HOLDER:?set HOLDER or ALLOC to a running holder}"
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
DATA=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination
CAPTURE=${CAPTURE:-$DATA/surrogate_capture_2708}
BASE=${BASE:-$DATA/surrogate_field_2708}
SCRIPT=analysis/gan_tuning/surrogate_feature_field_benchmark.py

cd "$ROOT"
NODE_EXPR=$(squeue -j "$HOLDER" -h -o %N)
mapfile -t NODES < <(scontrol show hostnames "$NODE_EXPR")
test "${#NODES[@]}" -gt 0
NODE_INDEX=${NODE_INDEX:-1}
if [ "$NODE_INDEX" -ge "${#NODES[@]}" ]; then
  NODE_INDEX=0
fi
NODE=${NODES[$NODE_INDEX]}
mkdir -p "$BASE/targets_fasthead" "$BASE/results_fasthead" logs
LOG=logs/surrogate_vgg_fasthead_h${HOLDER}_$(date +%Y%m%d_%H%M%S).log

echo "[VGG-FAST-FIELD] holder=$HOLDER node=$NODE head_lr=1e-3" | tee "$LOG"
srun --jobid="$HOLDER" --overlap --nodelist="$NODE" --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --gpu-bind=none bash -lc '
set -euo pipefail
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=. OMP_NUM_THREADS=8
CAPTURE='"$CAPTURE"'
BASE='"$BASE"'
SCRIPT='"$SCRIPT"'

# The three rotations are independent and fit on separate GPUs.
pids=()
for rotation in 0 1 2; do
  CUDA_VISIBLE_DEVICES="$rotation" python "$SCRIPT" extract-targets \
    --capture-dir "$CAPTURE" \
    --head "$BASE/head_curves/vgg_r${rotation}_lr1e3.pt" \
    --source vgg --rotation "$rotation" \
    --output "$BASE/targets_fasthead/vgg_r${rotation}_headlr1e3.npz" \
    > "$BASE/targets_fasthead/vgg_r${rotation}_headlr1e3.log" 2>&1 &
  pids+=("$!")
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
test "$rc" -eq 0

queue=()
launch_fit() {
  local gpu=$1 rotation=$2 mode=$3 condition=$4 lr=$5 tag=$6
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" fit-surrogate \
    --targets "$BASE/targets_fasthead/vgg_r${rotation}_headlr1e3.npz" \
    --teacher-mode "$mode" --condition "$condition" --lr "$lr" \
    --checkpoints 1,3,6,12,24,48,96,192,384 \
    --substeps-per-step 24 --seeds 3 \
    --output "$BASE/results_fasthead/vgg_r${rotation}_${mode}_${condition}_${tag}.json" \
    > "$BASE/results_fasthead/vgg_r${rotation}_${mode}_${condition}_${tag}.log" 2>&1 &
  queue+=("$!")
}
flush_fits() {
  local rc=0 p
  for p in "${queue[@]}"; do wait "$p" || rc=1; done
  queue=()
  test "$rc" -eq 0
}
for rotation in 0 1 2; do
  gpu=0
  for mode in online converged; do
    for condition in latent rgbmaxmin; do
      launch_fit "$gpu" "$rotation" "$mode" "$condition" 2e-4 lr2e4
      gpu=$((gpu+1))
    done
  done
  flush_fits
  gpu=0
  for mode in online converged; do
    for condition in latent rgbmaxmin; do
      launch_fit "$gpu" "$rotation" "$mode" "$condition" 1e-3 lr1e3
      gpu=$((gpu+1))
    done
  done
  flush_fits
done
' >> "$LOG" 2>&1

echo "[VGG-FAST-FIELD] complete results=$BASE/results_fasthead" | tee -a "$LOG"
