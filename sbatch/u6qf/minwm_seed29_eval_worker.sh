#!/bin/bash
# Run one of four disjoint evaluation shards on a four-GPU u6qf holder.
set -euo pipefail
: "${EVAL_SHARD:?0--3 shard index required}"

R=/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler
CODE="$R/ARRWM"
STAGE="$R/minwm_seed29/ARRWM"
PY="$R/miniforge3/envs/arrwm/bin/python3.10"
OUT="$CODE/grids/eval/out_minwm_seed29_sensitivity_20260920"
LOG="$OUT/logs/shard${EVAL_SHARD}"
mkdir -p "$LOG"
rm -f "$LOG/COMPLETE" "$LOG/FAILED"

export ARR="$STAGE"
export MINWM_SEED29_DIR="$STAGE/logs/eval_final/fleet30s/minwm_seed29"
export SEED65_DIR="$STAGE/analysis/eval_final/seed65_e1"
export FLEET30S_MODELS=minwm_seed29
export HF_HOME="$R/iclrv3_hf_cache"
export TORCH_HOME="$R/torch_home"
export PYTHONPATH="$CODE/grids/eval:$CODE/code_release:${PYTHONPATH:-}"
export OMP_NUM_THREADS=8

cd "$CODE"
pids=()
names=()

"$PY" grids/eval/final_v2_cpu.py --out "$OUT" --models minwm_seed29 \
  --shard-index "$EVAL_SHARD" --shard-count 4 >"$LOG/cpu.log" 2>&1 &
pids+=("$!"); names+=(cpu)

CUDA_VISIBLE_DEVICES=0 "$PY" grids/eval/final_v2_style.py \
  --out "$OUT" --models minwm_seed29 --shard-index "$EVAL_SHARD" --shard-count 4 \
  >"$LOG/style.log" 2>&1 &
pids+=("$!"); names+=(style)

CUDA_VISIBLE_DEVICES=1 "$PY" grids/eval/final_v2_control.py \
  --out "$OUT" --models minwm_seed29 --shard-index "$EVAL_SHARD" --shard-count 4 \
  >"$LOG/control.log" 2>&1 &
pids+=("$!"); names+=(control)

CUDA_VISIBLE_DEVICES=2 "$PY" grids/eval/final_v3_quality_gpu.py \
  --out "$OUT" --models minwm_seed29 --metric geometry \
  --shard-index "$EVAL_SHARD" --shard-count 4 >"$LOG/geometry.log" 2>&1 &
pids+=("$!"); names+=(geometry)

CUDA_VISIBLE_DEVICES=3 "$PY" grids/eval/final_v3_quality_gpu.py \
  --out "$OUT" --models minwm_seed29 --metric conjuration \
  --shard-index "$EVAL_SHARD" --shard-count 4 >"$LOG/conjuration.log" 2>&1 &
pids+=("$!"); names+=(conjuration)

rc=0
for i in "${!pids[@]}"; do
  wait "${pids[$i]}" || rc=1
done
if [ "$rc" -ne 0 ]; then
  touch "$LOG/FAILED"
  exit 1
fi
touch "$LOG/COMPLETE"
echo "EVAL_SHARD_COMPLETE shard=$EVAL_SHARD $(date -Is)"
