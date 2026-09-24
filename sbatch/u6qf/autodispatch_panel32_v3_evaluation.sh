#!/bin/bash
# Dispatch the corrected v3 panel evaluation into existing four-GPU holders.
# Generation is already immutable in definitive_final_eval_videos.  Metric
# producers validate hashes and therefore reuse only byte-identical v2 clips.
set -u

R=${ARRWM_REMOTE_ROOT:-/lus/lfs1aip2/scratch/u6qf/as1748.u6qf/ARRWM_straggler}
A=${ARRWM_CODE_ROOT:-$R/ARRWM}
P=${PANEL32_STAGE:-$R/panel32_v3_stage}
L="$R/panel32_stage/logs/holders"
OUT="$P/eval_final"
STATE=${PANEL32_EVAL_DISPATCH_STATE:-$P/eval_dispatch}
SHARDS=48
LOG="$STATE/autodispatch.log"
SOURCE_AUDIT=${PANEL32_SOURCE_CONJURATION_AUDIT:-$R/panel32_v2_stage/eval_final/conjuration_v2_audit}
mkdir -p "$STATE/locks" "$STATE/markers" "$L"
exec 9>"$STATE/autodispatch.lock"
flock -n 9 || exit 0

holder_ids() {
  awk '/PANEL32_HOLDER_READY/ {
         for (i=1; i<=NF; i++) if ($i ~ /^job=/) {
           sub(/^job=/, "", $i); print $i
         }
       }' "$L"/*.out 2>/dev/null | sort -u
}

idle_holders() {
  local holder running
  running=$(squeue -h -u "$(id -un)" -t R -n panel32-hold -o '%A')
  for holder in $(holder_ids); do
    grep -Fxq "$holder" <<<"$running" || continue
    test ! -e "$L/.holder_cmd_${holder}.sh" || continue
    test ! -e "$L/.holder_running_${holder}.sh" || continue
    echo "$holder"
  done
}

phase_complete() {
  local task=$1 batch=${2:-}
  case "$task" in
    setup) test -f "$OUT/logs/SETUP_COMPLETE" ;;
    reuse-conj) test -f "$STATE/markers/CONJURATION_REUSE_COMPLETE" ;;
    preflight)
      local lane shard
      for lane in 0 1 2 3; do
        shard=$((batch * 4 + lane))
        test -f "$OUT/markers/preflight_shard${shard}.COMPLETE" || return 1
      done ;;
    preflight-finalize) test -f "$OUT/preflight/ACTION_PREFLIGHT_COMPLETE" ;;
    conjuration-v2)
      local lane shard
      for lane in 0 1 2 3; do
        shard=$((batch * 4 + lane))
        test -f "$OUT/conjuration_v2_audit/markers/shard${shard}.COMPLETE" || return 1
      done ;;
    cpu|style|control|geometry|conjuration|longreloc)
      local lane shard
      for lane in 0 1 2 3; do
        shard=$((batch * 4 + lane))
        test -f "$OUT/markers/${task}_shard${shard}.COMPLETE" || return 1
      done ;;
    *) return 1 ;;
  esac
}

task_name() { printf '%s' "$1${2:+_$2}" | tr ':' '_'; }

release_stale() {
  local lock owner name
  for lock in "$STATE"/locks/*.lock; do
    test -d "$lock" || continue
    name=$(basename "$lock" .lock)
    test -f "$STATE/markers/${name}.COMPLETE" && continue
    owner=$(cat "$lock/holder" 2>/dev/null || true)
    if [ -z "$owner" ] || ! squeue -j "$owner" -h -t R -o '%i' 2>/dev/null | grep -q .; then
      rm -rf "$lock"
      echo "$(date -Is) release_stale name=$name owner=${owner:-none}" >> "$LOG"
    fi
  done
}

assign() {
  local holder=$1 task=$2 batch=${3:-} spec body name lock command temporary
  name=$(task_name "$task" "$batch")
  lock="$STATE/locks/${name}.lock"
  mkdir "$lock" 2>/dev/null || return 1
  printf '%s\n' "$holder" > "$lock/holder"
  if [ "$task" = reuse-conj ]; then
    body="'$R/miniforge3/envs/arrwm/bin/python' '$A/grids/eval/panel32_conjuration_v2_reuse.py' --manifest '$OUT/video_manifest.csv' --source-audit '$SOURCE_AUDIT' --output-audit '$OUT/conjuration_v2_audit' --report '$OUT/conjuration_v2_audit/reuse_report.json'; touch '$STATE/markers/CONJURATION_REUSE_COMPLETE'"
  else
    case "$task" in
      setup|preflight-finalize) spec="eval:$task" ;;
      conjuration-v2) spec="eval:conjuration-audit-v2:$batch" ;;
      *) spec="eval:$task:$batch" ;;
    esac
    body="export PANEL32_STAGE='$P'; bash '$A/sbatch/u6qf/panel32_v3_eval_sequence.sh' '$spec'"
  fi
  command="$L/.holder_cmd_${holder}.sh"
  temporary="$L/.holder_cmd_${holder}.tmp.$$"
  cat >"$temporary" <<EOF
#!/bin/bash
set -euo pipefail
trap 'rc=\$?; if [ "\$rc" -ne 0 ]; then rm -rf "$lock"; fi' EXIT
$body
touch "$STATE/markers/${name}.COMPLETE"
EOF
  chmod +x "$temporary"
  mv "$temporary" "$command"
  echo "$(date -Is) holder=$holder task=$task batch=${batch:-none}" >> "$LOG"
}

all_batches_complete() {
  local task=$1 batch
  for batch in $(seq 0 11); do phase_complete "$task" "$batch" || return 1; done
}

while true; do
  release_stale
  if phase_complete setup; then
    if all_batches_complete preflight && ! phase_complete preflight-finalize; then
      tasks="preflight-finalize"
    elif phase_complete preflight-finalize; then
      tasks=""
      if phase_complete reuse-conj; then
        for batch in $(seq 0 11); do
          phase_complete conjuration-v2 "$batch" || tasks="$tasks conjuration-v2:$batch"
        done
      fi
      for metric in geometry conjuration control style cpu longreloc; do
        for batch in $(seq 0 11); do
          phase_complete "$metric" "$batch" || tasks="$tasks $metric:$batch"
        done
      done
    else
      tasks="reuse-conj"
      for batch in $(seq 0 11); do
        phase_complete preflight "$batch" || tasks="$tasks preflight:$batch"
      done
    fi
  else
    tasks="setup"
  fi

  all_done=1
  for metric in geometry conjuration control style cpu longreloc conjuration-v2; do
    all_batches_complete "$metric" || { all_done=0; break; }
  done
  if [ "$all_done" -eq 1 ] && phase_complete preflight-finalize; then
    touch "$STATE/markers/METRICS_AND_AUDIT_COMPLETE"
    echo "$(date -Is) metrics_and_audit_complete" >> "$LOG"
    exit 0
  fi

  for holder in $(idle_holders); do
    claim="$L/.v3_dispatch_claim_${holder}.lock"
    mkdir "$claim" 2>/dev/null || continue
    assigned=0
    for token in $tasks; do
      task=${token%%:*}
      batch=""
      [ "$token" = "$task" ] || batch=${token#*:}
      phase_complete "$task" "$batch" && continue
      assign "$holder" "$task" "$batch" && { assigned=1; break; }
    done
    rm -rf "$claim"
    [ "$assigned" -eq 0 ] || true
  done
  sleep 20
done
