#!/bin/bash
# Watch the two 8-node full runs to completion (200 steps).
# lowt has occasional benign slow patches (video decode + IO) that recover, so the
# STALL threshold is relaxed to 1500s (25 min) and only fires while RUNNING.
declare -A F
F[5290666]="FULL-FWD|af-stat-wave-dmd3-fwd|200"
F[5290667]="FULL-LOWT|af-stat-wave-dmd3-lowt|200"
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/master_status.txt
: > "$STATUS"
START=$(date +%s); HEARTBEAT=3600; i=0
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START))
  line="[m$i +${EL}s]"; problem=""; alldone=1
  for id in 5290666 5290667; do
    IFS='|' read -r nm base goal <<< "${F[$id]}"
    E="logs/${base}_${id}.err"; O="logs/${base}_${id}.out"
    ST=$(squeue -j $id -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST="GONE"
    STEP=$(grep -hoE "step=[0-9]+/[0-9]+" "$E" "$O" 2>/dev/null | grep -oE "^step=[0-9]+" | grep -oE "[0-9]+" | sort -n | tail -1)
    MAT=$(grep -hc "\[42F-MATCH\]" "$E" "$O" 2>/dev/null | paste -sd+ | bc 2>/dev/null)
    DN=$(grep -hc "Training complete" "$E" "$O" 2>/dev/null | paste -sd+ | bc 2>/dev/null)
    OOM=$(grep -hcE "CUDA out of memory|OutOfMemoryError" "$E" "$O" 2>/dev/null | paste -sd+ | bc 2>/dev/null)
    ERR=$(grep -hcE "Traceback|RuntimeError|ValueError|NotImplementedError|AssertionError|IndexError" "$E" 2>/dev/null)
    HG=$(grep -hcE "Watchdog caught|operation timed out|ProcessGroupNCCL.*[Tt]imed out" "$E" 2>/dev/null)
    MT=$(stat -c %Y "$E" 2>/dev/null || echo 0); IDLE=$((NOW-MT))
    line="$line | $nm:$ST ${STEP:-_}/$goal m=${MAT:-0} dn=${DN:-0} oom=${OOM:-0} err=${ERR:-0} hg=${HG:-0}"
    [ "$ST" != "GONE" ] && alldone=0
    [ "${OOM:-0}" -gt 0 ] 2>/dev/null && problem="FULL_OOM:$nm($id)"
    [ "${ERR:-0}" -gt 0 ] 2>/dev/null && problem="FULL_ERR:$nm($id)"
    [ "${HG:-0}" -gt 0 ] 2>/dev/null && problem="FULL_HANG:$nm($id)"
    if [ "$ST" = "RUNNING" ] && [ "$MT" -gt 0 ] && [ "$IDLE" -gt 1500 ] && [ -n "${STEP:-}" ]; then problem="FULL_STALL>25min:$nm($id,step=${STEP})"; fi
  done
  echo "$line" >> "$STATUS"
  [ -n "$problem" ] && { echo "MASTER_PROBLEM $problem" >> "$STATUS"; break; }
  [ "$alldone" -eq 1 ] && { echo "MASTER_ALL_DONE (both full runs terminal)" >> "$STATUS"; break; }
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "MASTER_HEARTBEAT el=${EL}s" >> "$STATUS"; break; }
  sleep 120
done
