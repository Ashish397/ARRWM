#!/bin/bash
declare -A J=( [fwd800]=5355972 [nofwd800]=5355973 [fwd1000]=5356163 )
declare -A N=( [fwd800]=c8-sw-dmd3-fwd-ode800 [nofwd800]=c8-sw-dmd3-nofwd-ode800 [fwd1000]=c8-sw-dmd3-fwd-ode1000 )
ERRRE='Traceback|RuntimeError|KeyError|CUDA out of memory|OutOfMemory|ChildFailedError|size mismatch|NCCL.*(error|timeout)|Watchdog|AssertionError'
lg(){ echo "logs/${N[$1]}_${J[$1]}.err"; }
report(){ echo "=== $(date -u +%H:%M:%S) ==="; squeue -j ${J[fwd800]},${J[nofwd800]},${J[fwd1000]} -o "%.10i %.24j %.9T %.8M %R" 2>/dev/null; }
start=$(date +%s)
for i in $(seq 1 240); do
  done_ct=0
  for k in fwd800 nofwd800 fwd1000; do
    f=$(lg $k)
    e=$(grep -hcE "$ERRRE" "$f" 2>/dev/null | head -1)
    if [ "${e:-0}" -gt 0 ] 2>/dev/null; then oom=$(grep -hcE "OutOfMemory" "$f" 2>/dev/null|head -1); report; echo "ERROR in $k (${J[$k]}) oom=$oom:"; grep -hE "$ERRRE" "$f" 2>/dev/null | grep -viE "destroy_process_group" | tail -4; exit 1; fi
    st=$(squeue -j ${J[$k]} -h -o "%T" 2>/dev/null)
    if [ -z "$st" ]; then fin=$(sacct -j ${J[$k]} -n -o State 2>/dev/null|head -1|tr -d ' ')
      if echo "$fin"|grep -qE "FAILED|TIMEOUT|NODE_FAIL"; then report; echo "$k ENDED BADLY: $fin"; tail -6 "$f" 2>/dev/null; exit 1; fi
      echo "$fin"|grep -qE "COMPLETED|CANCELLED" && done_ct=$((done_ct+1)); continue; fi
    if [ "$st" = "RUNNING" ] && [ -f "$f" ]; then age=$(( $(date +%s) - $(stat -c %Y "$f") )); [ $age -gt 1500 ] && { report; echo "STALL $k idle ${age}s"; exit 2; }; fi
  done
  [ "$done_ct" -ge 3 ] && { report; echo "ALL 3 DMD RUNS FINISHED"; exit 0; }
  line=""; for k in fwd800 nofwd800 fwd1000; do s=$(grep -hoE "step[ =][0-9]+" "$(lg $k)" 2>/dev/null|tail -1|grep -oE "[0-9]+"); line="$line $k=${s:-_}"; done
  echo "[t+$((($(date +%s)-start)/60))m] step:$line /200"
  sleep 90
done
report; echo "window elapsed"
