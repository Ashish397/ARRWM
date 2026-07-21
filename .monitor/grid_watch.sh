#!/bin/bash
# Stall watcher for grid sweeps 5694068/5694069 (768 mp4s each expected under
# logs/eval_final/G/{pca8_8node,16node}/control_test) + the 7 FL jobs if still
# queued. Alerts (exit 2) on 10 min of no progress while running.
cd /scratch/u6ex/as1748.u6ex/ARRWM
JOBS=5694068,5694069
last=-1 stall=0
while true; do
  sq=$(squeue -j $JOBS -h -o '%T' 2>/dev/null)
  n=$(ls logs/eval_final/G/*/control_test/*_raw.mp4 2>/dev/null | wc -l)
  nrun=$(echo "$sq" | grep -c RUNNING)
  echo "$(date +%H:%M) grid videos=$n/1536 running=$nrun pending=$(echo "$sq" | grep -c PENDING)"
  if [ -z "$sq" ]; then
    echo "grid jobs left the queue: videos=$n/1536"
    sacct -j $JOBS --format=JobID,JobName%14,State,Elapsed -n | grep -v '\.'
    exit 0
  fi
  if [ "$nrun" -gt 0 ]; then
    if [ "$n" = "$last" ]; then stall=$((stall+1)); else stall=0; fi
    if [ $stall -ge 2 ]; then echo "ALERT: grid jobs stalled at $n videos for 10+ min"; exit 2; fi
  fi
  last=$n
  sleep 300
done
