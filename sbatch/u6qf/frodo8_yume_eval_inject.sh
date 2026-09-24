#!/bin/bash
# Inject an evaluation payload into an existing four-GPU holder allocation.
# Usage: frodo8_yume_eval_inject.sh MODE ALLOC [NODE] [SHARD_COUNT]
# MODE is setup, preflight, or a numeric metric shard index.
set -euo pipefail

MODE=${1:?usage: frodo8_yume_eval_inject.sh MODE ALLOC [NODE] [SHARD_COUNT]}
ALLOC=${2:?usage: frodo8_yume_eval_inject.sh MODE ALLOC [NODE] [SHARD_COUNT]}
NODE=${3:-}
SHARDS=${4:-4}
case "$ALLOC" in *[!0-9]*|'') echo "ALLOC must be a raw numeric allocation ID" >&2; exit 2 ;; esac
case "$SHARDS" in *[!0-9]*|'') echo "SHARD_COUNT must be numeric" >&2; exit 2 ;; esac
if [ "$SHARDS" -lt 1 ]; then echo "SHARD_COUNT must be positive" >&2; exit 2; fi
case "$MODE" in
  setup|preflight) ;;
  *[!0-9]*|'') echo "MODE must be setup, preflight, or a numeric shard" >&2; exit 2 ;;
esac

STATE=$(squeue -h -j "$ALLOC" -o '%T' | head -1)
if [ "$STATE" != RUNNING ]; then
  echo "allocation $ALLOC is not RUNNING (state=${STATE:-not-in-squeue})" >&2
  exit 1
fi
if [ -z "$NODE" ]; then NODE=$(squeue -h -j "$ALLOC" -o '%N' | head -1); fi
if [ -z "$NODE" ] || [ "$NODE" = "(null)" ] || [[ "$NODE" == *","* ]] || [[ "$NODE" == *"["* ]] || [[ "$NODE" == *"]"* ]] || [[ "$NODE" == *" "* ]]; then
  echo "could not resolve one explicit node for allocation $ALLOC: $NODE" >&2
  exit 1
fi

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
CODE=${ARRWM_CODE_ROOT:-$R/ARRWM}
export ARRWM_REMOTE_ROOT="$R" ARRWM_CODE_ROOT="$CODE" AF_ROOT="$CODE"
export FRODO8_EVAL_SHARDS="$SHARDS"
case "$MODE" in
  setup) PAYLOAD="$CODE/sbatch/u6qf/frodo8_yume_eval_setup.sh"; ARGS=() ;;
  preflight) PAYLOAD="$CODE/sbatch/u6qf/frodo8_yume_action_preflight_on_holder.sh"; ARGS=() ;;
  *) PAYLOAD="$CODE/sbatch/u6qf/frodo8_yume_eval_on_4gpu_holder.sh"; ARGS=("$MODE" "$SHARDS") ;;
esac
test -x "$PAYLOAD"

LOCK_ROOT="$R/aligned32_stage/locks"
mkdir -p "$LOCK_ROOT"
exec 8>"$LOCK_ROOT/frodo8_eval_phase.lock"
if [ "$MODE" = setup ] || [ "$MODE" = preflight ]; then
  flock -n -x 8 || { echo "another evaluation phase is active" >&2; exit 75; }
else
  flock -n -s 8 || { echo "setup or preflight is active" >&2; exit 75; }
  exec 9>"$LOCK_ROOT/frodo8_eval_shard_${MODE}.lock"
  flock -n -x 9 || { echo "metric shard $MODE is already active" >&2; exit 75; }
fi

echo "FRODO8_EVAL_INJECT mode=$MODE alloc=$ALLOC node=$NODE $(date -Is)"
exec srun --jobid="$ALLOC" --overlap --external-launcher \
  --nodes=1 --nodelist="$NODE" --ntasks=1 --kill-on-bad-exit=1 \
  "$PAYLOAD" "${ARGS[@]}"
