#!/bin/bash
# Watch the froz SMOKE (5296993, decay_end=20 -> exercises the freeze in 40
# steps) as a gate for the froz FULL run (5296994, 8 nodes, decay_end=50).
# If the smoke trips a hard problem, cancel the full before it burns node-hours.
# Counts summed across .err/.out with paste|bc (grep -hc on 2 files = per-file).
SMOKE=5296993; SBASE=smoke-dmd3-fwd-froz
FULL=5296994;  FBASE=af-stat-wave-dmd3-fwd-froz
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/froz_status.txt
: > "$STATUS"
SSEEN=0; FSEEN=0
START=$(date +%s); HEARTBEAT=3600; i=0
csum() { grep -hcE "$1" "$2" "$3" 2>/dev/null | paste -sd+ | bc 2>/dev/null; }
jstate() { local s; s=$(squeue -j "$1" -h -o "%T" 2>/dev/null); [ -z "$s" ] && echo GONE || echo "$s"; }
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START)); problem=""; alldone=1
  # --- smoke (gate) ---
  SE="logs/${SBASE}_${SMOKE}.err"; SO="logs/${SBASE}_${SMOKE}.out"
  SST=$(jstate $SMOKE); [ "$SST" != GONE ] && SSEEN=1
  SSTEP=$(grep -hoE "step=[0-9]+/40" "$SE" "$SO" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  SDN=$(csum "Training complete" "$SE" "$SO")
  SOOM=$(csum "CUDA out of memory|OutOfMemoryError" "$SE" "$SO")
  SERR=$(grep -hcE "Traceback|RuntimeError|ValueError|AssertionError|IndexError|size mismatch" "$SE" 2>/dev/null)
  SHG=$(grep -hcE "Watchdog caught|operation timed out|ProcessGroupNCCL.*[Tt]imed out" "$SE" 2>/dev/null)
  SMT=$(stat -c %Y "$SE" 2>/dev/null || echo 0); SIDLE=$((NOW-SMT))
  [ "$SST" != GONE ] || [ "$SSEEN" -eq 0 ] && alldone=0
  { [ "${SOOM:-0}" -gt 0 ] || { [ "${SERR:-0}" -gt 0 ] && [ "${SDN:-0}" -eq 0 ]; } || [ "${SHG:-0}" -gt 0 ]; } 2>/dev/null && problem="SMOKE_FAIL->cancel_full($FULL)"
  if [ "$SST" = RUNNING ] && [ "$SMT" -gt 0 ] && [ "$SIDLE" -gt 1500 ] && [ -n "${SSTEP:-}" ]; then problem="SMOKE_STALL->cancel_full($FULL)"; fi
  # --- full (report) ---
  FE="logs/${FBASE}_${FULL}.err"; FO="logs/${FBASE}_${FULL}.out"
  FST=$(jstate $FULL); [ "$FST" != GONE ] && FSEEN=1
  FSTEP=$(grep -hoE "step=[0-9]+/200" "$FE" "$FO" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  FDN=$(csum "Training complete" "$FE" "$FO")
  FOOM=$(csum "CUDA out of memory|OutOfMemoryError" "$FE" "$FO")
  FERR=$(grep -hcE "Traceback|RuntimeError|ValueError|AssertionError|IndexError|size mismatch" "$FE" 2>/dev/null)
  [ "$FST" != GONE ] || [ "$FSEEN" -eq 0 ] && alldone=0
  { [ "${FOOM:-0}" -gt 0 ] || { [ "${FERR:-0}" -gt 0 ] && [ "${FDN:-0}" -eq 0 ]; }; } 2>/dev/null && problem="FULL_FAIL:$FULL"
  echo "[m$i +${EL}s] SMOKE:$SST ${SSTEP:-_}/40 dn=${SDN:-0} oom=${SOOM:-0} err=${SERR:-0} hg=${SHG:-0} | FULL:$FST ${FSTEP:-_}/200 dn=${FDN:-0} oom=${FOOM:-0} err=${FERR:-0}" >> "$STATUS"
  [ -n "$problem" ] && { echo "FROZ_PROBLEM $problem" >> "$STATUS"; break; }
  if [ "$SST" = GONE ] && [ "$SSEEN" -eq 1 ] && [ "${SDN:-0}" -gt 0 ] && [ "$(grep -c SMOKE_GREEN "$STATUS")" -eq 0 ]; then
    echo "FROZ_SMOKE_GREEN (smoke complete @40, no errors)" >> "$STATUS"
  fi
  [ "$alldone" -eq 1 ] && { echo "FROZ_ALL_DONE (smoke+full terminal)" >> "$STATUS"; break; }
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "FROZ_HEARTBEAT el=${EL}s" >> "$STATUS"; break; }
  sleep 120
done
