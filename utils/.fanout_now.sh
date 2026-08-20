#!/bin/bash
# Wrapper: kill any stale fan-out watcher, then arm a fresh one on the
# current canary and record which job it is watching (the previous watcher's
# target could not be confirmed from ps, and a watcher pointed at a dead job
# aborts silently).
pkill -f fanout_dmd_arms 2>/dev/null
sleep 2
CANARY=$(squeue --me -h -n dmd10k-mse -o '%i' | head -1)
if [ -z "$CANARY" ]; then echo "[fanout-now] no dmd10k-mse in queue"; exit 1; fi
echo "[fanout-now] arming on canary $CANARY at $(date)"
cd /scratch/u6ex/as1748.u6ex/ARRWM
CANARY=$CANARY nohup bash utils/.fanout_dmd_arms.sh >> logs/fanout_dmd2.log 2>&1 &
sleep 2
echo "[fanout-now] armed; watcher pid $(pgrep -f fanout_dmd_arms | head -1)"
