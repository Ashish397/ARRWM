#!/bin/bash
# Watch ODE-F training (5323219). Catches: clean_only data load (paired count),
# dual-CD enabled, startup smoke, early steps + the new metrics, errors/OOM/NaN.
ID=5328758; BASE=ode-F
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/ode_f_status.txt; : > "$STATUS"; SEEN=0; SAW_RUN=0
START=$(date +%s); HEARTBEAT=10800; i=0
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START)); problem=""
  E="logs/${BASE}_${ID}.err"; O="logs/${BASE}_${ID}.out"
  ST=$(squeue -j $ID -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST=GONE
  [ "$ST" != GONE ] && SEEN=1
  PAIRED=$(grep -hoE "PairedTrajectoryDataset: [0-9]+ paired" "$E" "$O" 2>/dev/null | grep -oE "[0-9]+" | head -1)
  CDEN=$(grep -hc "dual-CD ENABLED" "$E" "$O" 2>/dev/null)
  STEP=$(grep -hoE "step[ =][0-9]+" "$E" "$O" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  OOM=$(grep -hcE "CUDA out of memory|OutOfMemoryError" "$E" 2>/dev/null)
  NAN=$(grep -hcE "nan|NaN|Inf" "$E" 2>/dev/null)
  ERR=$(grep -hcE "Traceback|RuntimeError|ValueError|KeyError|AssertionError|exhausted all" "$E" 2>/dev/null)
  echo "[m$i +${EL}s] $ST paired=${PAIRED:-_} cd_en=${CDEN:-0} step=${STEP:-_}/1000 oom=${OOM:-0} nan=${NAN:-0} err=${ERR:-0}" >> "$STATUS"
  if [ "$ST" = RUNNING ] && [ "$SAW_RUN" -eq 0 ]; then SAW_RUN=1; echo "ODE_F_STARTED" >> "$STATUS"; fi
  [ "${OOM:-0}" -gt 0 ] 2>/dev/null && problem="OOM"
  [ "${ERR:-0}" -gt 0 ] 2>/dev/null && problem="ERR"
  [ -n "$problem" ] && { echo "ODE_F_PROBLEM $problem (step=${STEP:-_})" >> "$STATUS"; grep -hE "Traceback|Error|exhausted" "$E" 2>/dev/null | tail -4 >> "$STATUS"; break; }
  if [ "$ST" = GONE ] && [ "$SEEN" -eq 1 ]; then
    DN=$(grep -hcE "training (done|complete)|ODE-F training done" "$O" "$E" 2>/dev/null)
    [ "${STEP:-0}" -ge 990 ] 2>/dev/null && echo "ODE_F_GREEN (reached ${STEP}/1000)" >> "$STATUS" || echo "ODE_F_GONE (step=${STEP:-_}/1000 dn=${DN:-0})" >> "$STATUS"
    break
  fi
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "ODE_F_HEARTBEAT $ST step=${STEP:-_}/1000" >> "$STATUS"; break; }
  sleep 90
done
