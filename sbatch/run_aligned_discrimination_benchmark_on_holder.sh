#!/bin/bash
# Run the frozen-feature extraction and GAN-budget head tests after the exact
# boundary capture has completed.  This consumes one node / four GPUs inside
# an existing holder and never submits or cancels a Slurm job.
set -euo pipefail

: "${HOLDER:?set HOLDER to a running two-node holder}"
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
DATA_ROOT=${DATA_ROOT:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination}
CAPTURE_DIR=${CAPTURE_DIR:-$DATA_ROOT/capture_2708}
FEATURE_DIR=${FEATURE_DIR:-$DATA_ROOT/features_2708}
RESULT_DIR=${RESULT_DIR:-$DATA_ROOT/results_2708}

cd "$ROOT"
NODE_EXPR=$(squeue -j "$HOLDER" -h -o %N)
test -n "$NODE_EXPR" && test "$NODE_EXPR" != "(null)"
NODE=$(scontrol show hostnames "$NODE_EXPR" | head -1)
mkdir -p "$FEATURE_DIR" "$RESULT_DIR" logs
STAMP=$(date +%Y%m%d_%H%M%S)
LOG=logs/gan_aligned_discrimination_h${HOLDER}_${STAMP}.log

echo "[ALIGNED-BENCH] holder=$HOLDER node=$NODE capture=$CAPTURE_DIR" | tee "$LOG"
srun --jobid="$HOLDER" --overlap --nodelist="$NODE" --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --gpu-bind=none bash -lc '
set -euo pipefail
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=. OMP_NUM_THREADS=8
SCRIPT=analysis/gan_tuning/gan_aligned_discrimination.py
CAPTURE_DIR='"$CAPTURE_DIR"'
FEATURE_DIR='"$FEATURE_DIR"'
RESULT_DIR='"$RESULT_DIR"'

python "$SCRIPT" audit --capture-dir "$CAPTURE_DIR"

CUDA_VISIBLE_DEVICES=0 python "$SCRIPT" extract --capture-dir "$CAPTURE_DIR" \
  --feature-dir "$FEATURE_DIR" --source dinov2 --batch-size 8 \
  > "$RESULT_DIR/extract_dinov2.log" 2>&1 & p0=$!
CUDA_VISIBLE_DEVICES=1 python "$SCRIPT" extract --capture-dir "$CAPTURE_DIR" \
  --feature-dir "$FEATURE_DIR" --source vgg --batch-size 4 \
  > "$RESULT_DIR/extract_vgg.log" 2>&1 & p1=$!
CUDA_VISIBLE_DEVICES=2 python "$SCRIPT" extract --capture-dir "$CAPTURE_DIR" \
  --feature-dir "$FEATURE_DIR" --source rn50 --batch-size 4 \
  > "$RESULT_DIR/extract_rn50.log" 2>&1 & p2=$!
CUDA_VISIBLE_DEVICES=3 python "$SCRIPT" extract --capture-dir "$CAPTURE_DIR" \
  --feature-dir "$FEATURE_DIR" --source pixgan --batch-size 8 \
  > "$RESULT_DIR/extract_pixgan.log" 2>&1 & p3=$!

rc=0
wait "$p0" || rc=1
wait "$p1" || rc=1
wait "$p2" || rc=1
wait "$p3" || rc=1
if [ "$rc" -ne 0 ]; then
  for s in dinov2 vgg rn50 pixgan; do
    echo "===== extract_$s ====="
    tail -80 "$RESULT_DIR/extract_$s.log" || true
  done
  exit 21
fi

export CUDA_VISIBLE_DEVICES=""
python "$SCRIPT" evaluate --feature-dir "$FEATURE_DIR" --source dinov2 \
  --output "$RESULT_DIR/dinov2.json" > "$RESULT_DIR/evaluate_dinov2.log" 2>&1 & e0=$!
python "$SCRIPT" evaluate --feature-dir "$FEATURE_DIR" --source vgg \
  --output "$RESULT_DIR/vgg.json" > "$RESULT_DIR/evaluate_vgg.log" 2>&1 & e1=$!
python "$SCRIPT" evaluate --feature-dir "$FEATURE_DIR" --source rn50 \
  --output "$RESULT_DIR/rn50.json" > "$RESULT_DIR/evaluate_rn50.log" 2>&1 & e2=$!
python "$SCRIPT" evaluate --feature-dir "$FEATURE_DIR" --source pixgan \
  --output "$RESULT_DIR/pixgan.json" > "$RESULT_DIR/evaluate_pixgan.log" 2>&1 & e3=$!
rc=0
wait "$e0" || rc=1
wait "$e1" || rc=1
wait "$e2" || rc=1
wait "$e3" || rc=1
if [ "$rc" -ne 0 ]; then
  for s in dinov2 vgg rn50 pixgan; do
    echo "===== evaluate_$s ====="
    tail -80 "$RESULT_DIR/evaluate_$s.log" || true
  done
  exit 22
fi

python - <<PY
import json
from pathlib import Path
p=Path("$RESULT_DIR")
for s in ("dinov2","vgg","rn50","pixgan"):
 d=json.load(open(p/f"{s}.json"))
 lin=d["linear_upper_bound"]["k2f5"]["row_mean_auc"]["median"]
 live=d["online_head"]["k2f5"]["live_lr_captured_steps_once"]["row_mean_auc"]["median"]
 live30=d["online_head"]["k2f5"]["live_lr_30updates_cycled_evidence"]["row_mean_auc"]["median"]
 five=d["online_head"]["k2f5"]["live_lr_5headsteps_cached_evidence"]["row_mean_auc"]["median"]
 lr10=d["online_head"]["k2f5"]["10x_lr_captured_steps_once"]["row_mean_auc"]["median"]
 perm=d["controls"]["within_ride_label_permutation_k2f5"]["row_mean_auc"]["median"]
 print(f"{s:8s} linear={lin:.3f} live={live:.3f} live30={live30:.3f} 5head={five:.3f} lr10={lr10:.3f} perm={perm:.3f}")
PY
' >> "$LOG" 2>&1

echo "[ALIGNED-BENCH] complete results=$RESULT_DIR" | tee -a "$LOG"
