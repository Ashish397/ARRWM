#!/bin/bash
# Run two full-chain exact-residual ladder rungs, one per node, inside an
# existing two-node holder. The holder allocation itself is never cancelled.
set -euo pipefail

: "${HOLDER:?set HOLDER to a running two-node holder}"
: "${LEFT_RUNG:?set LEFT_RUNG}"
: "${RIGHT_RUNG:?set RIGHT_RUNG}"

ROOT=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
BASE=${BASE:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/local_vjp_granular_audit_r0_h6158256}
PAIRED=${PAIRED:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/paired_decoder_pullback_r0_h6158256/paired_vgg_r0.npz}
CARTESIAN=${CARTESIAN:-/scratch/u6ex/as1748.u6ex/ARRWM_data/gan_aligned_discrimination/surrogate_field_2708/decoder_pullback_r0_vgg4_h6156714/vgg_r0_live_pullback.npz}
OUT_ROOT=${OUT_ROOT:-$BASE/exact_residual_ladder_h${HOLDER}}

cd "$ROOT"
NODE_EXPR=$(squeue -j "$HOLDER" -h -o %N)
test -n "$NODE_EXPR" && test "$NODE_EXPR" != "(null)"
mapfile -t NODES < <(scontrol show hostnames "$NODE_EXPR")
if [ "${#NODES[@]}" -ne 2 ]; then
  echo "expected two nodes, found ${#NODES[@]}: $NODE_EXPR" >&2
  exit 2
fi
mkdir -p "$OUT_ROOT"

run_rung() {
  local node=$1 rung=$2 stages
  case "$rung" in
    exact13) stages=5,6,7,9,10,11,13 ;;
    exact13_2) stages=2,5,6,7,9,10,11,13 ;;
    exact13_2_3) stages=2,3,5,6,7,9,10,11,13 ;;
    all_residual) stages=1,2,3,5,6,7,9,10,11,13,14,15 ;;
    *) echo "unknown rung $rung" >&2; return 3 ;;
  esac
  srun --jobid="$HOLDER" --overlap --nodelist="$node" --nodes=1 --ntasks=1 \
    --gpus-per-node=4 --gpu-bind=none bash -lc '
set -euo pipefail
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/etc/profile.d/conda.sh
conda activate arrwm
cd '"$ROOT"'
export PYTHONPATH=. OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
BASE='"$BASE"'
PAIRED='"$PAIRED"'
CARTESIAN='"$CARTESIAN"'
OUT='"$OUT_ROOT"'/'"$rung"'
STAGES='"$stages"'
mkdir -p "$OUT"
COMMON=(--generic-results "$BASE/results_chain_generic"
  --anchored-results "$BASE/results_chain_identity"
  --checkpoints "$BASE/chain_checkpoints")

pids=()
CUDA_VISIBLE_DEVICES=0 python analysis/gan_tuning/decoder_shaped_chain_benchmark.py \
  --paired-bank "$PAIRED" "${COMMON[@]}" --examples 81 --seeds 3 \
  --exact-fixed-stages --exact-residual-stages "$STAGES" \
  --output "$OUT/paired81.json" >"$OUT/paired81.log" 2>&1 &
pids+=("$!")
for shard in 0 1 2; do
  gpu=$((shard + 1))
  CUDA_VISIBLE_DEVICES=$gpu python analysis/gan_tuning/cartesian_chain_transfer_benchmark.py \
    --cartesian-bank "$CARTESIAN" "${COMMON[@]}" --seeds 3 \
    --family fixed_vgg_head --z-role test --v-role test --positive-only \
    --shard-index "$shard" --num-shards 3 \
    --exact-residual-stages "$STAGES" \
    --output "$OUT/cartesian_shard${shard}.json" \
    >"$OUT/cartesian_shard${shard}.log" 2>&1 &
  pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
if [ "$rc" -ne 0 ]; then
  for log in "$OUT"/*.log; do echo "===== $log ====="; tail -120 "$log" || true; done
  exit 4
fi
CUDA_VISIBLE_DEVICES="" python analysis/gan_tuning/summarize_exact_residual_ladder.py rung \
  --rung '"$rung"' --exact-stages "$STAGES" --paired "$OUT/paired81.json" \
  --cartesian-dir "$OUT" --cartesian-shards 3 --output "$OUT/summary.json" \
  >"$OUT/summary.log" 2>&1
date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT/COMPLETE"
' >"$OUT_ROOT/${rung}_${node}.holder.log" 2>&1
}

echo "[exact-ladder] holder=$HOLDER ${NODES[0]}=$LEFT_RUNG ${NODES[1]}=$RIGHT_RUNG"
run_rung "${NODES[0]}" "$LEFT_RUNG" & left_pid=$!
run_rung "${NODES[1]}" "$RIGHT_RUNG" & right_pid=$!
rc=0
wait "$left_pid" || rc=1
wait "$right_pid" || rc=1
if [ "$rc" -ne 0 ]; then
  tail -160 "$OUT_ROOT"/*.holder.log || true
  exit 5
fi
date -u +%Y-%m-%dT%H:%M:%SZ >"$OUT_ROOT/COMPLETE"
