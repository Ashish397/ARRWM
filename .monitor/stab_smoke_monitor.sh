#!/bin/bash
# Watch the af-stat-wave-dmd3-fwd-stab SMOKE (2 nodes). On clean completion the
# full 8-node run can be submitted. lowt-style slow patches recover, so STALL
# threshold relaxed to 1500s and only fires while RUNNING.
ID=5301066
BASE=smoke-dmd3-fwd-stab
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/stab_smoke_status.txt
: > "$STATUS"
SEEN=0
START=$(date +%s); HEARTBEAT=3600; i=0
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START)); problem=""
  E="logs/${BASE}_${ID}.err"; O="logs/${BASE}_${ID}.out"
  ST=$(squeue -j $ID -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST="GONE"
  [ "$ST" != "GONE" ] && SEEN=1
  STEP=$(grep -hoE "step=[0-9]+/40" "$E" "$O" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  MAT=$(grep -hc "\[42F-MATCH\]" "$E" 2>/dev/null)
  DN=$(grep -hc "Training complete" "$E" "$O" 2>/dev/null)
  OOM=$(grep -hcE "CUDA out of memory|OutOfMemoryError" "$E" 2>/dev/null)
  ERR=$(grep -hcE "Traceback|RuntimeError|ValueError|NotImplementedError|AssertionError|IndexError|size mismatch" "$E" 2>/dev/null)
  HG=$(grep -hcE "Watchdog caught|operation timed out|ProcessGroupNCCL.*[Tt]imed out" "$E" 2>/dev/null)
  MT=$(stat -c %Y "$E" 2>/dev/null || echo 0); IDLE=$((NOW-MT))
  echo "[m$i +${EL}s] STAB-SMOKE:$ST ${STEP:-_}/40 m=${MAT:-0} dn=${DN:-0} oom=${OOM:-0} err=${ERR:-0} hg=${HG:-0} idle=${IDLE}s" >> "$STATUS"
  [ "${OOM:-0}" -gt 0 ] 2>/dev/null && problem="OOM"
  [ "${ERR:-0}" -gt 0 ] 2>/dev/null && problem="ERR"
  [ "${HG:-0}" -gt 0 ] 2>/dev/null && problem="HANG"
  if [ "$ST" = "RUNNING" ] && [ "$MT" -gt 0 ] && [ "$IDLE" -gt 1500 ] && [ -n "${STEP:-}" ]; then problem="STALL>25min(step=${STEP})"; fi
  [ -n "$problem" ] && { echo "STAB_PROBLEM $problem" >> "$STATUS"; break; }
  if [ "$ST" = "GONE" ] && [ "$SEEN" -eq 1 ]; then
    if [ "${DN:-0}" -gt 0 ] && [ "${ERR:-0}" -eq 0 ]; then
      echo "STAB_SMOKE_GREEN (Training complete, no errors -> submit full run)" >> "$STATUS"
    else
      echo "STAB_SMOKE_GONE_NO_COMPLETE (terminated without 'Training complete' -> investigate)" >> "$STATUS"
    fi
    break
  fi
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "STAB_HEARTBEAT el=${EL}s" >> "$STATUS"; break; }
  sleep 120
done
