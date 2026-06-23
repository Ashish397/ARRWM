#!/bin/bash
J=5356442; N=c8-stat-freal-trad; F=logs/${N}_${J}.err
ERRRE='Traceback|RuntimeError|KeyError|CUDA out of memory|OutOfMemory|ChildFailedError|size mismatch|NCCL.*(error|timeout)|Watchdog|AssertionError'
report(){ echo "=== $(date -u +%H:%M:%S) ==="; squeue -j $J -o "%.10i %.22j %.9T %.8M %R" 2>/dev/null; }
start=$(date +%s)
for i in $(seq 1 200); do
  e=$(grep -hcE "$ERRRE" "$F" 2>/dev/null | head -1)
  [ "${e:-0}" -gt 0 ] 2>/dev/null && { report; echo "ERROR:"; grep -hE "$ERRRE" "$F" 2>/dev/null | grep -viE "destroy_process_group" | tail -6; exit 1; }
  st=$(squeue -j $J -h -o "%T" 2>/dev/null)
  if [ -z "$st" ]; then fin=$(sacct -j $J -n -o State 2>/dev/null|head -1|tr -d ' '); report; echo "FINISHED: $fin"; echo "$fin"|grep -qE "FAILED|TIMEOUT|NODE_FAIL" && { tail -6 "$F" 2>/dev/null; exit 1; }; exit 0; fi
  if [ "$st" = "RUNNING" ] && [ -f "$F" ]; then age=$(( $(date +%s) - $(stat -c %Y "$F") )); [ $age -gt 1500 ] && { report; echo "STALL idle ${age}s"; exit 2; }; fi
  step=$(grep -hoE "step[ =][0-9]+" "$F" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  echo "[t+$((($(date +%s)-start)/60))m] step=${step:-_}/200"
  sleep 90
done
report; echo "window elapsed"
