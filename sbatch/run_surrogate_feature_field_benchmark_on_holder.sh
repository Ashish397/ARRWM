#!/bin/bash
# Exact-boundary feature-teacher -> latent-gradient-surrogate benchmark.
# Runs inside an existing holder; never submits or cancels an allocation.
set -euo pipefail

HOLDER=${HOLDER:-${ALLOC:-}}
: "${HOLDER:?set HOLDER or ALLOC to a running holder}"
ROOT=/scratch/u6ex/as1748.u6ex/ARRWM
DATA=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination
CAPTURE=${CAPTURE:-$DATA/surrogate_capture_2708}
FEATURES=${FEATURES:-$DATA/features_2708}
OUT=${OUT:-$DATA/surrogate_field_2708}
SCRIPT=analysis/gan_tuning/surrogate_feature_field_benchmark.py

cd "$ROOT"
NODE_EXPR=$(squeue -j "$HOLDER" -h -o %N)
test -n "$NODE_EXPR" && test "$NODE_EXPR" != "(null)"
NODE=$(scontrol show hostnames "$NODE_EXPR" | head -1)
mkdir -p "$OUT/heads" "$OUT/head_curves" "$OUT/targets" "$OUT/results" logs
LOG=logs/surrogate_feature_field_h${HOLDER}_$(date +%Y%m%d_%H%M%S).log

echo "[SURROGATE-FIELD] holder=$HOLDER node=$NODE capture=$CAPTURE" | tee "$LOG"
srun --jobid="$HOLDER" --overlap --nodelist="$NODE" --nodes=1 --ntasks=1 \
  --gpus-per-node=4 --gpu-bind=none bash -lc '
set -euo pipefail
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=. OMP_NUM_THREADS=8
DATA=/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination
CAPTURE='"$CAPTURE"'
FEATURES='"$FEATURES"'
OUT='"$OUT"'
SCRIPT=analysis/gan_tuning/surrogate_feature_field_benchmark.py

# The latent bank is useful only when every rank/step record completed.
python analysis/gan_tuning/gan_aligned_discrimination.py audit \
  --capture-dir "$CAPTURE" > "$OUT/capture_audit.json"
python - <<PY
import numpy as np
from pathlib import Path
fs=sorted(Path("$CAPTURE").glob("*.npz"))
assert len(fs)==72, f"expected 72 complete records, got {len(fs)}"
for p in fs:
    with np.load(p, allow_pickle=False) as z:
        assert "real_latent" in z and "fake_latent" in z, p
        assert z["real_latent"].shape == z["fake_latent"].shape
print("latent capture audit PASS", len(fs))
PY

# Discriminator convergence/tracking heads. The production LR, the aligned
# test 10x remedy, and a faster bracket are all measured; 2e-4 is the
# predeclared teacher used for field targets.
pids=()
for source in dinov2 vgg rn50; do
  for rotation in 0 1 2; do
    OMP_NUM_THREADS=2 python "$SCRIPT" fit-head \
      --feature-dir "$FEATURES" --source "$source" \
      --rotation "$rotation" --updates 300 \
      --curve-updates 1,3,6,9,15,30,60,120,300 \
      --output "$OUT/heads/${source}_r${rotation}.pt" \
      > "$OUT/heads/${source}_r${rotation}.log" 2>&1 &
    pids+=("$!")
    OMP_NUM_THREADS=2 python "$SCRIPT" fit-head \
      --feature-dir "$FEATURES" --source "$source" \
      --rotation "$rotation" --updates 300 --lr 2e-5 \
      --curve-updates 1,3,6,9,15,30,60,120,300 \
      --output "$OUT/head_curves/${source}_r${rotation}_lr2e5.pt" \
      > "$OUT/head_curves/${source}_r${rotation}_lr2e5.log" 2>&1 &
    pids+=("$!")
    OMP_NUM_THREADS=2 python "$SCRIPT" fit-head \
      --feature-dir "$FEATURES" --source "$source" \
      --rotation "$rotation" --updates 300 --lr 1e-3 \
      --curve-updates 1,3,6,9,15,30,60,120,300 \
      --output "$OUT/head_curves/${source}_r${rotation}_lr1e3.pt" \
      > "$OUT/head_curves/${source}_r${rotation}_lr1e3.log" 2>&1 &
    pids+=("$!")
  done
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
test "$rc" -eq 0

# VAE gradient targets dominate runtime. One source per GPU, rotations
# sequential inside that source so only three VAE copies are resident.
pids=()
gpu=0
for source in dinov2 vgg rn50; do
  (
    for rotation in 0 1 2; do
      CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" extract-targets \
        --capture-dir "$CAPTURE" \
        --head "$OUT/heads/${source}_r${rotation}.pt" \
        --source "$source" --rotation "$rotation" \
        --output "$OUT/targets/${source}_r${rotation}.npz" \
        > "$OUT/targets/${source}_r${rotation}.log" 2>&1
    done
  ) & pids+=("$!")
  gpu=$((gpu+1))
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
if [ "$rc" -ne 0 ]; then
  tail -80 "$OUT"/targets/*.log || true
  exit 31
fi

# Student conditions. Four jobs at a time use the four GPUs; every job uses
# the production 96x6 predictor and three matched random seeds.
queue=()
launch_fit() {
  local gpu=$1 source=$2 rotation=$3 mode=$4 condition=$5 lr=$6 tag=$7
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" fit-surrogate \
    --targets "$OUT/targets/${source}_r${rotation}.npz" \
    --teacher-mode "$mode" --condition "$condition" --lr "$lr" \
    --checkpoints 1,3,6,12,24,48,96,192,384 \
    --substeps-per-step 24 --seeds 3 \
    --output "$OUT/results/${source}_r${rotation}_${mode}_${condition}_${tag}.json" \
    > "$OUT/results/${source}_r${rotation}_${mode}_${condition}_${tag}.log" 2>&1 &
  queue+=("$!")
}
flush_fits() {
  local rc=0 p
  for p in "${queue[@]}"; do wait "$p" || rc=1; done
  queue=()
  test "$rc" -eq 0
}
for source in dinov2 vgg rn50; do
  for rotation in 0 1 2; do
    gpu=0
    for mode in online converged; do
      for condition in latent rgbmaxmin; do
        launch_fit "$gpu" "$source" "$rotation" "$mode" "$condition" 2e-4 lr2e4
        gpu=$((gpu+1))
      done
    done
    flush_fits
    gpu=0
    for mode in online converged; do
      for condition in latent rgbmaxmin; do
        launch_fit "$gpu" "$source" "$rotation" "$mode" "$condition" 1e-3 lr1e3
        gpu=$((gpu+1))
      done
    done
    flush_fits
  done
done

python - <<PY
import json
from pathlib import Path
p=Path("$OUT/results")
for source in ("dinov2","vgg","rn50"):
  for mode in ("online","converged"):
    for cond in ("latent","rgbmaxmin"):
      for lr in ("lr2e4","lr1e3"):
        vals=[]
        for rot in range(3):
          d=json.load(open(p/f"{source}_r{rot}_{mode}_{cond}_{lr}.json"))
          if mode=="online":
            vals.append(d["aggregate"]["post_to_next_global_cosine"]["median"])
          else:
            vals.append(d["aggregate"]["384"]["test"]["global_cosine"]["median"])
        vals.sort()
        print(source, mode, cond, lr, "rotation_median", vals[1], "rotations", vals)
PY
' >> "$LOG" 2>&1

echo "[SURROGATE-FIELD] complete results=$OUT" | tee -a "$LOG"
