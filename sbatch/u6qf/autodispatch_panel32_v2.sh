#!/bin/bash
# Fill running four-GPU holders with independent v2 generation work.  One
# command is injected at a time; the holder remains reusable until its Slurm
# limit.  Corrected accepted-context conjuration is interleaved at roughly one
# batch per five generation actions so it cannot starve the critical rerun.
set -u

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=${PANEL32_STAGE:-$R/panel32_v2_stage}
L="$R/panel32_stage/logs/holders"
MANIFEST="$A/grids/eval/panel32_locked_v2.json"
PROVENANCE="$P/sources/panel32_source_provenance.json"
BUNDLES="$P/ours_seed_bundles/panel32_seed_bundles.json"
STATE="$P/dispatch"
LOG="$STATE/autodispatch.log"
mkdir -p "$STATE/markers" "$STATE/locks" "$L"
exec 9>"$STATE/autodispatch.lock"
flock -n 9 || exit 0

generation_tasks() {
  local action arm
  for arm in lingbot dreamx matrixgame2 minwm minwm_ode; do
    for action in F FR R BR B BL L FL N; do echo "external:$arm:$action"; done
  done
  for action in F FR R BR B BL L FL N; do echo "yume:$action"; done
  for arm in recoverybase nocarn nocommit noaux nogan meanenergy vartv; do
    for action in F FR R BR B BL L FL N; do echo "ours:$arm:$action"; done
  done
  for arm in local_kl pointwise_mse; do
    for action in F FR R BR B BL L FL N; do echo "ode:$arm:$action"; done
  done
}

holder_ids() {
  awk '/PANEL32_HOLDER_READY/ {
         for (i=1; i<=NF; i++) if ($i ~ /^job=/) {
           sub(/^job=/, "", $i); print $i
         }
       }' "$L"/*.out 2>/dev/null | sort -u
}

safe_name() { printf '%s' "$1" | tr ':' '_'; }
task_complete() { test -f "$STATE/markers/$(safe_name "$1").COMPLETE"; }
task_available() {
  case "$1" in
    ours:*|ode:*) test -f "$STATE/markers/seed_and_reuse.COMPLETE" ;;
    *) test -f "$STATE/markers/external_reuse.COMPLETE" ;;
  esac
}

release_stale_locks() {
  local lock owner task
  for lock in "$STATE"/locks/*.lock; do
    test -d "$lock" || continue
    task=$(basename "$lock" .lock)
    test -f "$STATE/markers/$task.COMPLETE" && continue
    owner=$(cat "$lock/holder" 2>/dev/null || true)
    if [ -z "$owner" ] || ! squeue -j "$owner" -h -t R -o '%i' 2>/dev/null | grep -q .; then
      rm -rf "$lock"
      echo "$(date -Is) release_stale task=$task owner=${owner:-none}" >> "$LOG"
    fi
  done
}

assign_command() {
  local holder=$1 task=$2 body=$3 lock=$4
  local command="$L/.holder_cmd_${holder}.sh"
  local temporary="$L/.holder_cmd_${holder}.tmp.$$"
  cat >"$temporary" <<EOF
#!/bin/bash
set -euo pipefail
trap 'rc=\$?; if [ "\$rc" -ne 0 ]; then rm -rf "$lock"; fi' EXIT
$body
touch "$STATE/markers/$(safe_name "$task").COMPLETE"
EOF
  chmod +x "$temporary"
  mv "$temporary" "$command"
  echo "$(date -Is) holder=$holder task=$task command=$command" >> "$LOG"
}

claim_task() {
  local holder=$1 task=$2 body=$3 name lock
  name=$(safe_name "$task")
  task_complete "$task" && return 1
  lock="$STATE/locks/$name.lock"
  mkdir "$lock" 2>/dev/null || return 1
  printf '%s\n' "$holder" > "$lock/holder"
  assign_command "$holder" "$task" "$body" "$lock"
}

conj_complete() {
  local batch=$1 lane shard audit
  audit="$R/panel32_stage/eval_final/conjuration_v2_audit_current"
  for lane in 0 1 2 3; do
    shard=$((batch * 4 + lane))
    test -f "$audit/markers/shard${shard}.COMPLETE" || return 1
  done
}

test -f "$MANIFEST" && test -f "$PROVENANCE" || {
  echo "missing v2 source contract" >&2; exit 1;
}
dispatch_count=0
while true; do
  release_stale_locks
  all_generation_done=1
  while read -r task; do task_complete "$task" || { all_generation_done=0; break; }; done < <(generation_tasks)
  all_conj_done=1
  for batch in $(seq 0 11); do conj_complete "$batch" || { all_conj_done=0; break; }; done
  if [ "$all_generation_done" -eq 1 ]; then
    touch "$STATE/markers/GENERATION_COMPLETE"
  fi
  if [ "$all_generation_done" -eq 1 ] && [ "$all_conj_done" -eq 1 ]; then
    touch "$STATE/markers/GENERATION_AND_CURRENT_CONJURATION_COMPLETE"
    echo "$(date -Is) generation_and_current_conjuration_complete" >> "$LOG"
    exit 0
  fi

  for holder in $(holder_ids); do
    squeue -j "$holder" -h -t R -o '%i' 2>/dev/null | grep -q . || continue
    test ! -e "$L/.holder_cmd_${holder}.sh" || continue
    test ! -e "$L/.holder_running_${holder}.sh" || continue
    claim="$L/.v2_dispatch_claim_${holder}.lock"
    mkdir "$claim" 2>/dev/null || continue

    assigned=0
    if [ ! -f "$STATE/markers/seed_and_reuse.COMPLETE" ]; then
      task=seed_and_reuse
      lock="$STATE/locks/$task.lock"
      if mkdir "$lock" 2>/dev/null; then
        printf '%s\n' "$holder" > "$lock/holder"
        body="PANEL32_STAGE='$P' PANEL32_BUNDLE_ROOT='$P/ours_seed_bundles' PANEL32_BUNDLE_LOG='$P/logs/ours_seed_bundles.log' bash '$A/sbatch/u6qf/panel32_seed_bundles_on_holder.sh' '$holder' '$MANIFEST' '$PROVENANCE'; '$R/miniforge3/envs/arrwm/bin/python' '$A/grids/eval/panel32_seed_bundles.py' validate --panel-manifest '$MANIFEST' --source-provenance '$PROVENANCE' --output '$P/ours_seed_bundles' --wan-model-root '$R/frodobots' --pca-checkpoint '$A/code_release/preprocessing/checkpoints/pca_basis.pt' --cotracker-repo '$R/torch_home/hub/facebookresearch_co-tracker_main'; '$R/miniforge3/envs/arrwm/bin/python' '$A/grids/eval/panel32_reuse_unchanged_v1.py' --old-root '$R/panel32_stage' --new-root '$P' --old-provenance '$R/panel32_stage/sources/panel32_source_provenance.json' --new-provenance '$PROVENANCE' --manifest '$MANIFEST' --phase all --bundle-index '$BUNDLES' > '$P/logs/reuse_all.json'"
        assign_command "$holder" "$task" "$body" "$lock"
        assigned=1
      fi
    fi

    # Reserve about one sixth of assignments for the accepted-context
    # conjuration audit while generation remains the dominant workload.
    if [ "$assigned" -eq 0 ] && [ $((dispatch_count % 6)) -eq 5 ]; then
      for batch in $(seq 0 11); do
        conj_complete "$batch" && continue
        task="conj_current_batch_$batch"
        body="'$A/sbatch/u6qf/panel32_conjuration_v2_current_batch_on_4gpu_holder.sh' '$batch' 48"
        claim_task "$holder" "$task" "$body" && { assigned=1; break; }
      done
    fi

    if [ "$assigned" -eq 0 ]; then
      while read -r task; do
        task_available "$task" || continue
        body="export PANEL32_STAGE='$P' PANEL32_MANIFEST='$MANIFEST' PANEL32_PROVENANCE='$PROVENANCE' PANEL32_BUNDLES='$BUNDLES'; bash '$A/sbatch/u6qf/panel32_holder_sequence.sh' '$task'"
        claim_task "$holder" "$task" "$body" && { assigned=1; break; }
      done < <(generation_tasks)
    fi

    if [ "$assigned" -eq 0 ]; then
      for batch in $(seq 0 11); do
        conj_complete "$batch" && continue
        task="conj_current_batch_$batch"
        body="'$A/sbatch/u6qf/panel32_conjuration_v2_current_batch_on_4gpu_holder.sh' '$batch' 48"
        claim_task "$holder" "$task" "$body" && { assigned=1; break; }
      done
    fi
    rm -rf "$claim"
    [ "$assigned" -eq 0 ] || dispatch_count=$((dispatch_count + 1))
  done
  sleep 20
done
