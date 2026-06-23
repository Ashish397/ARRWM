#!/bin/bash
# Watch both ODE-F runs: 5328758 (with dual-CD) + 5342377 (noCD ablation).
# Reports startup / paired-count / step / errors for each.
declare -A J=( [5328758]="ode-F" [5342377]="ode-F-noCD" )
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/ode_f_both_status.txt; : > "$STATUS"
declare -A SEEN SAWRUN
START=$(date +%s); HEARTBEAT=10800; i=0
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START)); line="[m$i +${EL}s]"; problem=""; alldone=1
  for id in 5328758 5342377; do
    base="${J[$id]}"; E="logs/${base}_${id}.err"; O="logs/${base}_${id}.out"
    ST=$(squeue -j $id -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST=GONE
    [ "$ST" != GONE ] && SEEN[$id]=1
    [ "$ST" = RUNNING ] && [ -z "${SAWRUN[$id]:-}" ] && { SAWRUN[$id]=1; echo "${base} STARTED" >> "$STATUS"; }
    PAIRED=$(grep -hoE "PairedTrajectoryDataset: [0-9]+ paired" "$E" "$O" 2>/dev/null | grep -oE "[0-9]+" | head -1)
    STEP=$(grep -hoE "step[ =][0-9]+" "$E" "$O" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
    ERR=$(grep -hcE "Traceback|RuntimeError|ValueError|KeyError|AssertionError|exhausted all|OutOfMemory" "$E" 2>/dev/null)
    line="$line | ${base}:$ST paired=${PAIRED:-_} step=${STEP:-_}/1000 err=${ERR:-0}"
    { [ "$ST" != GONE ] || [ -z "${SEEN[$id]:-}" ]; } && alldone=0
    { [ "${ERR:-0}" -gt 0 ] && [ "$ST" != GONE ]; } 2>/dev/null && problem="${base}_ERR($id)"
    # GONE-early (before ~step 980) with no done marker = failed
    if [ "$ST" = GONE ] && [ -n "${SEEN[$id]:-}" ] && [ "${STEP:-0}" -lt 980 ] 2>/dev/null && [ "${ERR:-0}" -gt 0 ]; then problem="${base}_FAILED($id step=${STEP:-_})"; fi
  done
  echo "$line" >> "$STATUS"
  [ -n "$problem" ] && { echo "ODE_F_BOTH_PROBLEM $problem" >> "$STATUS"; break; }
  [ "$alldone" -eq 1 ] && { echo "ODE_F_BOTH_DONE" >> "$STATUS"; break; }
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "ODE_F_BOTH_HEARTBEAT" >> "$STATUS"; break; }
  sleep 120
done
