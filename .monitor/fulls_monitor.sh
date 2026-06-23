#!/bin/bash
# Single monitor for BOTH stabilization full runs:
#   stab (5296659): decoupled teacher-t + v14 anchor + GAN-gate coupling
#   froz (5296994): same + teacher LR decays to 0 by step 50 then frozen
# Watches to completion; relaxed 25-min stall threshold; counts summed per-file.
declare -A F
F[5296659]="STAB|af-stat-wave-dmd3-fwd-stab"
F[5296994]="FROZ|af-stat-wave-dmd3-fwd-froz"
declare -A SEEN
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/fulls_status.txt
: > "$STATUS"
START=$(date +%s); HEARTBEAT=3600; i=0
csum() { grep -hcE "$1" "$2" "$3" 2>/dev/null | paste -sd+ | bc 2>/dev/null; }
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START)); line="[m$i +${EL}s]"; problem=""; alldone=1
  for id in 5296659 5296994; do
    IFS='|' read -r tag base <<< "${F[$id]}"
    E="logs/${base}_${id}.err"; O="logs/${base}_${id}.out"
    ST=$(squeue -j $id -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST="GONE"
    [ "$ST" != GONE ] && SEEN[$id]=1
    STEP=$(grep -hoE "step=[0-9]+/200" "$E" "$O" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
    DN=$(csum "Training complete" "$E" "$O")
    OOM=$(csum "CUDA out of memory|OutOfMemoryError" "$E" "$O")
    ERR=$(grep -hcE "Traceback|RuntimeError|ValueError|AssertionError|IndexError|size mismatch" "$E" 2>/dev/null)
    HG=$(grep -hcE "Watchdog caught|operation timed out|ProcessGroupNCCL.*[Tt]imed out" "$E" 2>/dev/null)
    MT=$(stat -c %Y "$E" 2>/dev/null || echo 0); IDLE=$((NOW-MT))
    line="$line | $tag:$ST ${STEP:-_}/200 dn=${DN:-0} oom=${OOM:-0} err=${ERR:-0} hg=${HG:-0}"
    { [ "$ST" != GONE ] || [ -z "${SEEN[$id]:-}" ]; } && alldone=0
    [ "${OOM:-0}" -gt 0 ] 2>/dev/null && problem="${tag}_OOM($id)"
    { [ "${ERR:-0}" -gt 0 ] && [ "${DN:-0}" -eq 0 ]; } 2>/dev/null && problem="${tag}_ERR($id)"
    [ "${HG:-0}" -gt 0 ] 2>/dev/null && problem="${tag}_HANG($id)"
    if [ "$ST" = RUNNING ] && [ "$MT" -gt 0 ] && [ "$IDLE" -gt 1500 ] && [ -n "${STEP:-}" ]; then problem="${tag}_STALL($id,step=${STEP})"; fi
  done
  echo "$line" >> "$STATUS"
  [ -n "$problem" ] && { echo "FULLS_PROBLEM $problem" >> "$STATUS"; break; }
  [ "$alldone" -eq 1 ] && { echo "FULLS_ALL_DONE (both full runs terminal)" >> "$STATUS"; break; }
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "FULLS_HEARTBEAT el=${EL}s" >> "$STATUS"; break; }
  sleep 120
done
