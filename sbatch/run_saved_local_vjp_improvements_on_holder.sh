#!/bin/bash
# Run two isolated no-recapture local-VJP experiments, one per node, inside an
# existing two-node/four-GPU holder. This script never submits or cancels jobs.
set -euo pipefail

: "${HOLDER:?set HOLDER to the running two-node holder job id}"
: "${LEFT_ARM:?set LEFT_ARM}"
: "${RIGHT_ARM:?set RIGHT_ARM}"

ROOT=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
BASE=${BASE:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256}
OUT_ROOT=${OUT_ROOT:-$BASE/improvements_no_recapture_h${HOLDER}}
BANK=$BASE/bank
SCRIPT=analysis/gan_tuning/saved_local_vjp_improvements.py

cd "$ROOT"
NODE_EXPR=$(squeue -j "$HOLDER" -h -o %N)
test -n "$NODE_EXPR" && test "$NODE_EXPR" != "(null)"
mapfile -t NODES < <(scontrol show hostnames "$NODE_EXPR")
if [ "${#NODES[@]}" -ne 2 ]; then
  echo "expected two holder nodes, found ${#NODES[@]}: $NODE_EXPR" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT" logs

run_node() {
  local node=$1 arm=$2
  srun --jobid="$HOLDER" --overlap --nodelist="$node" --nodes=1 --ntasks=1 \
    --gpus-per-node=4 --gpu-bind=none bash -lc '
set -euo pipefail
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/etc/profile.d/conda.sh
conda activate arrwm
cd '"$ROOT"'
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
BANK='"$BANK"'
OUT='"$OUT_ROOT"'/'"$arm"'
SCRIPT='"$SCRIPT"'
mkdir -p "$OUT/checkpoints"

pids=()
case '"$arm"' in
  high_anchor)
    specs=("13:24,48,72,0" "14:24,48,72,0" "15:24,48,72,0")
    for gpu in 0 1 2; do
      IFS=: read -r stage ranks <<<"${specs[$gpu]}"
      CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" screen --bank-dir "$BANK" \
        --stage "$stage" --ranks "$ranks" --output "$OUT/stage${stage}.json" \
        >"$OUT/stage${stage}.log" 2>&1 & pids+=("$!")
    done
    ;;
  low_anchor)
    specs=("1:48,96,192,288,0" "2:48,96,192,288,0" "3:48,96,192,288,0")
    for gpu in 0 1 2; do
      IFS=: read -r stage ranks <<<"${specs[$gpu]}"
      CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" screen --bank-dir "$BANK" \
        --stage "$stage" --ranks "$ranks" --output "$OUT/stage${stage}.json" \
        >"$OUT/stage${stage}.log" 2>&1 & pids+=("$!")
    done
    ;;
  high_correction)
    specs=("13:24" "13:48" "13:72")
    for gpu in 0 1 2; do
      IFS=: read -r stage rank <<<"${specs[$gpu]}"
      CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" fit-correction --bank-dir "$BANK" \
        --stage "$stage" --rank "$rank" --updates 600 \
        --output "$OUT/stage${stage}_rank${rank}.json" \
        --checkpoint-dir "$OUT/checkpoints" \
        >"$OUT/stage${stage}_rank${rank}.log" 2>&1 & pids+=("$!")
    done
    ;;
  low_correction)
    specs=("2:48" "2:96" "3:96" "3:192")
    for gpu in 0 1 2 3; do
      IFS=: read -r stage rank <<<"${specs[$gpu]}"
      CUDA_VISIBLE_DEVICES=$gpu python "$SCRIPT" fit-correction --bank-dir "$BANK" \
        --stage "$stage" --rank "$rank" --updates 600 \
        --output "$OUT/stage${stage}_rank${rank}.json" \
        --checkpoint-dir "$OUT/checkpoints" \
        >"$OUT/stage${stage}_rank${rank}.log" 2>&1 & pids+=("$!")
    done
    ;;
  *)
    echo "unknown no-recapture arm: '"$arm"'" >&2
    exit 3
    ;;
esac

rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  for log in "$OUT"/*.log; do echo "===== $log ====="; tail -100 "$log" || true; done
  exit 4
fi
CUDA_VISIBLE_DEVICES="" python "$SCRIPT" summarize --input-dir "$OUT" \
  --output "$OUT/arm_summary.json" >"$OUT/arm_summary.log" 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/COMPLETE"
' >"$OUT_ROOT/${arm}_${node}.holder.log" 2>&1
}

echo "[no-recapture] holder=$HOLDER ${NODES[0]}=$LEFT_ARM ${NODES[1]}=$RIGHT_ARM"
run_node "${NODES[0]}" "$LEFT_ARM" & left_pid=$!
run_node "${NODES[1]}" "$RIGHT_ARM" & right_pid=$!
rc=0
wait "$left_pid" || rc=1
wait "$right_pid" || rc=1
if [ "$rc" -ne 0 ]; then
  tail -120 "$OUT_ROOT"/*.holder.log || true
  exit 5
fi
date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT_ROOT/COMPLETE"
