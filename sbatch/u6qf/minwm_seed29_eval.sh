#!/bin/bash
# Run all five evaluation instruments for the isolated minWM 29-frame
# conditioning sensitivity inside an existing four-GPU u6qf holder.
set -euo pipefail

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
CODE="$R/ARRWM"
STAGE="$R/minwm_seed29/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
OUT="$CODE/grids/eval/out_minwm_seed29_sensitivity_20260920"
OLD="$CODE/grids/eval/out_iclr_final_v3_full_20260917"
LOG="$OUT/logs"
mkdir -p "$LOG"

export ARR="$STAGE"
export MINWM_SEED29_DIR="$STAGE/logs/eval_final/fleet30s/minwm_seed29"
export SEED65_DIR="$STAGE/analysis/eval_final/seed65_e1"
export FLEET30S_MODELS=minwm_seed29
export HF_HOME="$R/iclrv3_hf_cache"
export TORCH_HOME="$R/torch_home"
export PYTHONPATH="$CODE/grids/eval:$CODE/code_release:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

cd "$CODE"
"$PY" grids/eval/minwm_seed29_sensitivity.py manifest --out "$OUT" \
  >"$LOG/manifest.log" 2>&1

pids=()
names=()

"$PY" grids/eval/final_v2_cpu.py --out "$OUT" --models minwm_seed29 \
  >"$LOG/cpu.log" 2>&1 &
pids+=("$!"); names+=(cpu)

CUDA_VISIBLE_DEVICES=0 "$PY" grids/eval/final_v2_style.py \
  --out "$OUT" --models minwm_seed29 >"$LOG/style.log" 2>&1 &
pids+=("$!"); names+=(style)

CUDA_VISIBLE_DEVICES=1 "$PY" grids/eval/final_v2_control.py \
  --out "$OUT" --models minwm_seed29 >"$LOG/control.log" 2>&1 &
pids+=("$!"); names+=(control)

CUDA_VISIBLE_DEVICES=2 "$PY" grids/eval/final_v3_quality_gpu.py \
  --out "$OUT" --models minwm_seed29 --metric geometry \
  >"$LOG/geometry.log" 2>&1 &
pids+=("$!"); names+=(geometry)

CUDA_VISIBLE_DEVICES=3 "$PY" grids/eval/final_v3_quality_gpu.py \
  --out "$OUT" --models minwm_seed29 --metric conjuration \
  >"$LOG/conjuration.log" 2>&1 &
pids+=("$!"); names+=(conjuration)

rc=0
for i in "${!pids[@]}"; do
  if wait "${pids[$i]}"; then
    echo "${names[$i]} COMPLETE $(date -Is)" | tee -a "$LOG/status.log"
  else
    echo "${names[$i]} FAILED $(date -Is)" | tee -a "$LOG/status.log"
    rc=1
  fi
done
test "$rc" -eq 0

"$PY" grids/eval/minwm_seed29_sensitivity.py summarize \
  --out "$OUT" --old "$OLD" >"$LOG/summarize.log" 2>&1

echo "MINWM_SEED29_EVAL_COMPLETE $(date -Is)" | tee -a "$LOG/status.log"
