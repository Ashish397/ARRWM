#!/bin/bash
# Stall watcher for the 7 phase-FL jobs (48 mp4s expected per run under
# logs/eval_final/FL/<run>/control_test). Alerts (exit 2) if total video count
# is unchanged across 2 consecutive 5-min samples while any job RUNS; exits 0
# when all jobs leave the queue.
cd /scratch/u6ex/as1748.u6ex/ARRWM
JOBS=5693024,5693026,5693028,5693029,5693030,5693031,5693033
last=-1 stall=0
while true; do
  sq=$(squeue -j $JOBS -h -o '%T' 2>/dev/null)
  n=$(ls logs/eval_final/FL/*/control_test/*_raw.mp4 2>/dev/null | wc -l)
  nrun=$(echo "$sq" | grep -c RUNNING)
  echo "$(date +%H:%M) videos=$n/336 running=$nrun queued=$(echo "$sq" | grep -c PENDING)"
  if [ -z "$sq" ]; then
    echo "all FL jobs left the queue: videos=$n/336"
    sacct -j $JOBS --format=JobID,JobName%14,State,Elapsed -n | grep -v '\.'
    exit 0
  fi
  if [ "$nrun" -gt 0 ]; then
    if [ "$n" = "$last" ]; then stall=$((stall+1)); else stall=0; fi
    if [ $stall -ge 2 ]; then echo "ALERT: FL jobs stalled at $n videos for 10+ min with $nrun running"; exit 2; fi
  fi
  last=$n
  sleep 300
done
