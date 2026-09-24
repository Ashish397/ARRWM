#!/bin/bash
# Execute one or more evaluation phases inside an existing four-GPU holder.
# Arguments use eval:setup, eval:preflight, eval:TASK:BATCH, or eval:finish.
set -euo pipefail

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
CODE=${ARRWM_CODE_ROOT:-$R/ARRWM}
SHARDS=${PANEL32_EVAL_SHARDS:-48}
PANEL_ROOT=${PANEL32_STAGE_ROOT:-$R/panel32_stage}
LOCK_ROOT="$PANEL_ROOT/locks"
mkdir -p "$LOCK_ROOT"

run_phase() (
  set -euo pipefail
  local spec=${1:?evaluation phase required}
  local phase=${spec#eval:}
  local payload
  local -a args=()
  # The holder loop serialises command files, and this lock also excludes an
  # overlapping srun injected into the same allocation.
  if [ -n "${ALLOC:-}" ]; then
    exec 10>"$LOCK_ROOT/panel32_eval_alloc_${ALLOC}.lock"
    flock -n -x 10 || { echo "allocation $ALLOC already has a panel32 payload" >&2; exit 75; }
  fi
  case "$phase" in
    setup)
      payload="$CODE/sbatch/u6qf/panel32_eval_setup.sh" ;;
    matrix-strict)
      payload="$CODE/sbatch/u6qf/panel32_matrix_strict_validation.sh" ;;
    preflight|preflight-finalize)
      payload="$CODE/sbatch/u6qf/panel32_action_preflight_finalize.sh"
      args=("$SHARDS") ;;
    finish)
      payload="$CODE/sbatch/u6qf/panel32_eval_finish.sh" ;;
    conjuration-audit-v2:*)
      local batch=${phase#*:}
      case "$batch" in *[!0-9]*|'') echo "invalid batch in $spec" >&2; exit 2 ;; esac
      if [ "$batch" -ge $((SHARDS / 4)) ]; then
        echo "batch $batch is outside 0..$((SHARDS / 4 - 1))" >&2
        exit 2
      fi
      payload="$CODE/sbatch/u6qf/panel32_conjuration_v2_audit_batch_on_4gpu_holder.sh"
      args=("$batch" "$SHARDS")
      ;;
    conjuration-audit-kept-v2:*)
      local batch=${phase#*:}
      case "$batch" in *[!0-9]*|'') echo "invalid batch in $spec" >&2; exit 2 ;; esac
      if [ "$batch" -ge $((SHARDS / 4)) ]; then
        echo "batch $batch is outside 0..$((SHARDS / 4 - 1))" >&2
        exit 2
      fi
      payload="$CODE/sbatch/u6qf/panel32_conjuration_v2_kept_batch_on_4gpu_holder.sh"
      args=("$batch" "$SHARDS")
      ;;
    preflight:*|cpu:*|style:*|control:*|geometry:*|conjuration:*|conjuration-v2:*|longreloc:*)
      local task=${phase%%:*}
      local batch=${phase#*:}
      case "$batch" in *[!0-9]*|'') echo "invalid batch in $spec" >&2; exit 2 ;; esac
      if [ "$batch" -ge $((SHARDS / 4)) ]; then
        echo "batch $batch is outside 0..$((SHARDS / 4 - 1))" >&2
        exit 2
      fi
      if [ "$task" = preflight ]; then
        payload="$CODE/sbatch/u6qf/panel32_action_preflight_on_holder.sh"
        args=("$batch" "$SHARDS")
      else
        payload="$CODE/sbatch/u6qf/panel32_eval_metric_batch_on_4gpu_holder.sh"
        args=("$task" "$batch" "$SHARDS")
      fi
      ;;
    *) echo "invalid evaluation phase: $spec" >&2; exit 2 ;;
  esac
  test -x "$payload"

  exec 8>"$LOCK_ROOT/panel32_eval_exclusive.lock"
  if [ "$phase" = setup ] || [ "$phase" = preflight ] || \
     [ "$phase" = preflight-finalize ] || [ "$phase" = finish ]; then
    flock -n -x 8 || { echo "another evaluation phase is active" >&2; exit 75; }
  else
    flock -n -s 8 || { echo "setup, preflight, or finish is active" >&2; exit 75; }
    exec 9>"$LOCK_ROOT/panel32_eval_${phase/:/_}.lock"
    flock -n -x 9 || { echo "metric batch $phase is already active" >&2; exit 75; }
  fi
  export ARRWM_REMOTE_ROOT="$R" ARRWM_CODE_ROOT="$CODE" AF_ROOT="$CODE"
  export PANEL32_EVAL_SHARDS="$SHARDS"
  echo "PANEL32_EVAL_SEQUENCE_START phase=$phase holder=${ALLOC:-unknown} $(date -Is)"
  bash "$payload" "${args[@]}"
  echo "PANEL32_EVAL_SEQUENCE_DONE phase=$phase holder=${ALLOC:-unknown} $(date -Is)"
)

for specification in "$@"; do
  run_phase "$specification"
done
