#!/bin/bash
# Dispatch twelve four-GPU VLM triage batches to idle retained holders.
set -euo pipefail

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=${PANEL32_STAGE:-$R/panel32_v2_stage}
L="$R/panel32_stage/logs/holders"
DEST="$P/eval_final/conjuration_v2_audit/vlm_triage"
SHARDS=48
EXCLUDE_HOLDERS=${EXCLUDE_HOLDERS:-}
mkdir -p "$DEST/dispatch" "$DEST/markers"

holder_ids() {
  awk '/PANEL32_HOLDER_READY/ {
         for (i=1; i<=NF; i++) if ($i ~ /^job=/) {
           sub(/^job=/, "", $i); print $i
         }
       }' "$L"/*.out 2>/dev/null | sort -nu
}

is_excluded() {
  local holder=$1 item
  for item in $EXCLUDE_HOLDERS; do [ "$holder" = "$item" ] && return 0; done
  return 1
}

running=$(squeue -h -u "$(id -un)" -t R -n panel32-hold -o '%A' | sort -u)
batch=0
for holder in $(holder_ids); do
  [ "$batch" -lt 12 ] || break
  grep -Fxq "$holder" <<<"$running" || continue
  is_excluded "$holder" && continue
  test ! -e "$L/.holder_cmd_${holder}.sh" || continue
  test ! -e "$L/.holder_running_${holder}.sh" || continue
  complete=1
  for lane in 0 1 2 3; do
    shard=$((batch * 4 + lane))
    test -f "$DEST/markers/shard${shard}.COMPLETE" || complete=0
  done
  if [ "$complete" -eq 1 ]; then
    batch=$((batch + 1))
    continue
  fi
  command="$L/.holder_cmd_${holder}.sh"
  temporary="$L/.holder_cmd_${holder}.tmp.$$"
  printf '%s\n' \
    '#!/bin/bash' \
    'set -euo pipefail' \
    "export PANEL32_STAGE='$P' PANEL32_STAGE_ROOT='$P'" \
    "export PANEL32_EVAL_CONFIG='$P/panel32_eval_config.json'" \
    "export PANEL32_EVAL_OUT='$P/eval_final'" \
    "export PANEL32_LOCKED_MANIFEST='$A/grids/eval/panel32_locked_v2.json'" \
    "bash '$A/sbatch/u6qf/panel32_conjuration_v2_vlm_batch_on_4gpu_holder.sh' '$batch' '$SHARDS'" \
    >"$temporary"
  chmod +x "$temporary"
  mv "$temporary" "$command"
  printf '%s holder=%s batch=%s\n' "$(date -Is)" "$holder" "$batch" \
    >>"$DEST/dispatch/assignments.log"
  batch=$((batch + 1))
done

if [ "$batch" -ne 12 ]; then
  echo "needed 12 idle holders, assigned only $batch" >&2
  exit 1
fi
echo "CONJURATION_V2_VLM_TRIAGE_DISPATCHED batches=12 shards=$SHARDS $(date -Is)"
