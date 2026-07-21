#!/bin/bash
# Stall watcher for the noadaln jobs: sweep 5689208 (256 mp4s expected in
# logs/eval_final/S/noadaln/control_test) + cm 5689209 (csv rows growing).
# Alerts (exit 2) if a RUNNING job makes no progress for 2 consecutive 5-min
# samples; exits 0 when both jobs leave the queue.
cd /scratch/u6ex/as1748.u6ex/ARRWM
SW=5689208 CM=5689209
last_v=-1 last_r=-1 v_stall=0 c_stall=0
while true; do
  sq=$(squeue -j $SW,$CM -h -o '%i %T' 2>/dev/null)
  swst=$(echo "$sq" | awk -v j=$SW '$1==j{print $2}')
  cmst=$(echo "$sq" | awk -v j=$CM '$1==j{print $2}')
  nv=$(ls logs/eval_final/S/noadaln/control_test/*_raw.mp4 2>/dev/null | wc -l)
  nr=$(wc -l < analysis/chunk_metrics_noadaln.csv 2>/dev/null || echo 0)
  echo "$(date +%H:%M) sweep=$swst v=$nv | cm=$cmst rows=$nr"
  if [ -z "$sq" ]; then
    echo "both jobs left the queue: sweep videos=$nv cm rows=$nr"
    sacct -j $SW,$CM --format=JobID,JobName%16,State,Elapsed -n | grep -v '\.'
    exit 0
  fi
  if [ "$swst" = "RUNNING" ]; then
    if [ "$nv" = "$last_v" ]; then v_stall=$((v_stall+1)); else v_stall=0; fi
    if [ $v_stall -ge 2 ]; then echo "ALERT: sweep $SW stalled at $nv videos for 10+ min"; exit 2; fi
  fi
  if [ "$cmst" = "RUNNING" ]; then
    if [ "$nr" = "$last_r" ]; then c_stall=$((c_stall+1)); else c_stall=0; fi
    if [ $c_stall -ge 2 ]; then echo "ALERT: cm $CM stalled at $nr rows for 10+ min"; exit 2; fi
  fi
  last_v=$nv; last_r=$nr
  sleep 300
done
