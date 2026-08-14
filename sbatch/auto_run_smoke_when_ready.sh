#!/bin/bash
# Detached watcher: fire the rollout smoke the INSTANT an allocation lands.
#
# WHY: agent sessions get torn down, and anything in the session's process
# group dies with them (that is how allocations 6001818 / 6004082 were
# CANCELLED after 9.5h without ever running). Launch this with `setsid` and it
# outlives the session, so a node landing at 03:00 is used at 03:00 instead of
# whenever someone next looks.
#
# Watches every candidate allocation and drives the first one that starts.
#   setsid nohup bash sbatch/auto_run_smoke_when_ready.sh <jobid> [jobid...] \
#     > logs/auto_smoke_watch.log 2>&1 < /dev/null &
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
JOBS=("$@")
[ ${#JOBS[@]} -gt 0 ] || { echo "usage: $0 <jobid> [jobid...]"; exit 2; }
echo "[watch] $(date) watching: ${JOBS[*]}"

while true; do
  for j in "${JOBS[@]}"; do
    st=$(squeue -j "$j" -h -o "%T" 2>/dev/null)
    if [ "$st" = "RUNNING" ]; then
      echo "[watch] $(date) job $j RUNNING on $(squeue -j "$j" -h -o '%N') -- firing smoke"
      # STEPS=3: enough to prove the backward survives checkpoint recompute and
      # to time a step; not so many that a broken run burns the allocation.
      ALLOC="$j" TAG="auto$j" STEPS=3 bash sbatch/iact_roll_smoke.sh
      rc=$?
      echo "[watch] $(date) smoke exit=$rc  log=logs/iact_roll_auto${j}.log"
      grep -aE "CheckpointError|Traceback|RuntimeError|AssertionError|TRAIN-FAIL" \
        "logs/iact_roll_auto${j}.log" 2>/dev/null | head -5
      exit $rc
    fi
  done
  # Every candidate gone (finished/cancelled) => nothing left to wait for.
  alive=0
  for j in "${JOBS[@]}"; do
    squeue -j "$j" -h -o "%T" 2>/dev/null | grep -q . && alive=1
  done
  [ "$alive" = "1" ] || { echo "[watch] $(date) no candidate jobs left; exiting"; exit 3; }
  sleep 20
done
