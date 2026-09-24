#!/bin/bash
# Inject one panel32 evaluation phase into an existing four-GPU holder.
# Usage:
#   panel32_eval_inject.sh setup ALLOC [NODE] [SHARD_COUNT]
#   panel32_eval_inject.sh preflight:BATCH ALLOC [NODE] [SHARD_COUNT]
#   panel32_eval_inject.sh preflight-finalize ALLOC [NODE] [SHARD_COUNT]
#   panel32_eval_inject.sh finish ALLOC [NODE] [SHARD_COUNT]
#   panel32_eval_inject.sh TASK:BATCH ALLOC [NODE] [SHARD_COUNT]
set -euo pipefail

MODE=${1:?mode required}
ALLOC=${2:?allocation id required}
NODE=${3:-}
SHARDS=${4:-48}
if [[ ! "$ALLOC" =~ ^[0-9]+(_[0-9]+)?$ ]]; then
  echo "ALLOC must be a Slurm job id or array-element id (for example 6823827_47)" >&2
  exit 2
fi
case "$SHARDS" in *[!0-9]*|'') echo "SHARD_COUNT must be numeric" >&2; exit 2 ;; esac
if [ "$SHARDS" -lt 4 ] || [ $((SHARDS % 4)) -ne 0 ]; then
  echo "SHARD_COUNT must be a positive multiple of four and at least four" >&2
  exit 2
fi

STATE=$(squeue -h -j "$ALLOC" -o '%T' | head -1)
if [ "$STATE" != RUNNING ]; then
  echo "allocation $ALLOC is not RUNNING (state=${STATE:-not-in-squeue})" >&2
  exit 1
fi
if [ -z "$NODE" ]; then NODE=$(squeue -h -j "$ALLOC" -o '%N' | head -1); fi
if [ -z "$NODE" ] || [ "$NODE" = "(null)" ] || [[ "$NODE" == *","* ]] || \
   [[ "$NODE" == *"["* ]] || [[ "$NODE" == *"]"* ]] || [[ "$NODE" == *" "* ]]; then
  echo "could not resolve one explicit node for allocation $ALLOC: $NODE" >&2
  exit 1
fi

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
CODE=${ARRWM_CODE_ROOT:-$R/ARRWM}
export ARRWM_REMOTE_ROOT="$R" ARRWM_CODE_ROOT="$CODE" AF_ROOT="$CODE"
export PANEL32_EVAL_SHARDS="$SHARDS"
case "$MODE" in
  setup)
    PAYLOAD="$CODE/sbatch/u6qf/panel32_eval_setup.sh"; ARGS=() ;;
  preflight|preflight-finalize)
    PAYLOAD="$CODE/sbatch/u6qf/panel32_action_preflight_finalize.sh"; ARGS=("$SHARDS") ;;
  finish)
    PAYLOAD="$CODE/sbatch/u6qf/panel32_eval_finish.sh"; ARGS=() ;;
  preflight:*|cpu:*|style:*|control:*|geometry:*|conjuration:*|conjuration-v2:*|longreloc:*)
    TASK=${MODE%%:*}
    BATCH=${MODE#*:}
    case "$BATCH" in *[!0-9]*|'') echo "invalid batch in $MODE" >&2; exit 2 ;; esac
    if [ "$BATCH" -ge $((SHARDS / 4)) ]; then
      echo "batch $BATCH is outside 0..$((SHARDS / 4 - 1))" >&2
      exit 2
    fi
    if [ "$TASK" = preflight ]; then
      PAYLOAD="$CODE/sbatch/u6qf/panel32_action_preflight_on_holder.sh"
      ARGS=("$BATCH" "$SHARDS")
    else
      PAYLOAD="$CODE/sbatch/u6qf/panel32_eval_metric_batch_on_4gpu_holder.sh"
      ARGS=("$TASK" "$BATCH" "$SHARDS")
    fi
    ;;
  *) echo "invalid MODE=$MODE" >&2; exit 2 ;;
esac
test -x "$PAYLOAD"

LOCK_ROOT="$R/panel32_stage/locks"
mkdir -p "$LOCK_ROOT"
# Prevent two otherwise-disjoint modes from both binding lanes 0--3 on the
# same four-GPU holder.  Task locks alone cannot enforce one process per GPU.
exec 10>"$LOCK_ROOT/panel32_eval_alloc_${ALLOC}.lock"
flock -n -x 10 || { echo "allocation $ALLOC already has a panel32 payload" >&2; exit 75; }
if [ "$MODE" = setup ] || [ "$MODE" = preflight ] || \
   [ "$MODE" = preflight-finalize ] || [ "$MODE" = finish ]; then
  exec 8>"$LOCK_ROOT/panel32_eval_exclusive.lock"
  flock -n -x 8 || { echo "another exclusive evaluation phase is active" >&2; exit 75; }
else
  exec 8>"$LOCK_ROOT/panel32_eval_exclusive.lock"
  flock -n -s 8 || { echo "setup, preflight, or finish is active" >&2; exit 75; }
  exec 9>"$LOCK_ROOT/panel32_eval_${MODE/:/_}.lock"
  flock -n -x 9 || { echo "metric batch $MODE is already active" >&2; exit 75; }
fi

echo "PANEL32_EVAL_INJECT mode=$MODE alloc=$ALLOC node=$NODE shards=$SHARDS $(date -Is)"
exec srun --jobid="$ALLOC" --overlap --external-launcher \
  --nodes=1 --nodelist="$NODE" --ntasks=1 --kill-on-bad-exit=1 \
  "$PAYLOAD" "${ARGS[@]}"
