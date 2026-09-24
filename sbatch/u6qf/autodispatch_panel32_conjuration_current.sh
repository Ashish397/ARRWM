#!/bin/bash
# Login-node dispatcher for the queued panel32 holder arrays. It claims at
# most the 12 four-shard batches needed by the current accepted-context audit.
# It never cancels a holder and never overwrites another queued/running command.
set -u

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
L="$R/panel32_stage/logs/holders"
A="$R/ARRWM"
AUDIT="$R/panel32_stage/eval_final/conjuration_v2_audit_current"
DISPATCH_SCOPE=${PANEL32_DISPATCH_SCOPE:-current_conjuration_dispatch}
HOLDER_ARRAY=${PANEL32_HOLDER_ARRAY:-}
DISPATCH_LOG="$L/${DISPATCH_SCOPE}.log"
mkdir -p "$L" "$AUDIT/markers"
exec 9>"$L/${DISPATCH_SCOPE}.lock"
flock -n 9 || exit 0

holder_ids() {
  # Array notation in squeue is not necessarily SLURM_JOB_ID. The READY line
  # is emitted by the holder itself and records the exact command-file key.
  local logs=("$L"/*.out)
  if [ -n "$HOLDER_ARRAY" ]; then
    logs=("$L"/"${HOLDER_ARRAY}"_*.out)
  fi
  awk '/PANEL32_HOLDER_READY/ {
         for (i=1; i<=NF; i++) if ($i ~ /^job=/) {
           sub(/^job=/, "", $i); print $i
         }
       }' "${logs[@]}" 2>/dev/null | sort -u
}

batch_complete() {
  local batch=$1 lane shard
  for lane in 0 1 2 3; do
    shard=$((batch * 4 + lane))
    test -f "$AUDIT/markers/shard${shard}.COMPLETE" || return 1
  done
}

while true; do
  for batch in $(seq 0 11); do
    lock="$L/.current_conj_batch_${batch}.lock"
    test -d "$lock" || continue
    batch_complete "$batch" && continue
    owner=$(cat "$lock/holder" 2>/dev/null || true)
    if [ -n "$owner" ] && ! squeue -j "$owner" -h -t R -o '%i' 2>/dev/null | grep -q .; then
      rm -rf "$lock"
      echo "$(date -Is) released stale batch=$batch owner=$owner" >> "$DISPATCH_LOG"
    fi
  done

  all_done=1
  for batch in $(seq 0 11); do
    batch_complete "$batch" || { all_done=0; break; }
  done
  if [ "$all_done" -eq 1 ]; then
    echo "$(date -Is) all 48 current-context shards complete" >> "$DISPATCH_LOG"
    exit 0
  fi

  for holder in $(holder_ids); do
    squeue -j "$holder" -h -t R -o '%i' 2>/dev/null | grep -q . || continue
    test ! -e "$L/.holder_cmd_${holder}.sh" || continue
    test ! -e "$L/.holder_running_${holder}.sh" || continue

    for batch in $(seq 0 11); do
      batch_complete "$batch" && continue
      lock="$L/.current_conj_batch_${batch}.lock"
      mkdir "$lock" 2>/dev/null || continue
      printf '%s\n' "$holder" > "$lock/holder"
      command="$L/.holder_cmd_${holder}.sh"
      temporary="$L/.holder_cmd_${holder}.tmp.$$"
      printf '#!/bin/bash\n"%s/sbatch/u6qf/panel32_conjuration_v2_current_batch_on_4gpu_holder.sh" %s 48 || { rc=$?; rm -rf "%s"; exit "$rc"; }\n' \
        "$A" "$batch" "$lock" > "$temporary"
      chmod +x "$temporary"
      mv "$temporary" "$command"
      printf '%s holder=%s batch=%s command=%s\n' \
        "$(date -Is)" "$holder" "$batch" "$command" >> "$DISPATCH_LOG"
      break
    done
  done
  sleep 30
done
