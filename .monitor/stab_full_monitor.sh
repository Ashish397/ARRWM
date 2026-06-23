#!/bin/bash
# Watch the af-stat-wave-dmd3-fwd-stab FULL run (8 nodes, 200 steps). Tests the
# three stabilization fixes (decoupled teacher-t shift=1, v14 anchor 0.003,
# GAN-gate coupling) against the fwd baseline. Relaxed 25-min stall threshold.
# NOTE: counts are summed across .err/.out with paste|bc (grep -hc on two files
# prints one count PER FILE, which breaks -gt tests -- the smoke-monitor bug).
ID=5296659
BASE=af-stat-wave-dmd3-fwd-stab
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/stab_full_status.txt
: > "$STATUS"
SEEN=0
START=$(date +%s); HEARTBEAT=3600; i=0
csum() { grep -hcE "$1" "$2" "$3" 2>/dev/null | paste -sd+ | bc 2>/dev/null; }
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START)); problem=""
  E="logs/${BASE}_${ID}.err"; O="logs/${BASE}_${ID}.out"
  ST=$(squeue -j $ID -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST="GONE"
  [ "$ST" != "GONE" ] && SEEN=1
  STEP=$(grep -hoE "step=[0-9]+/200" "$E" "$O" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
  MAT=$(csum "\[42F-MATCH\]" "$E" "$O")
  DN=$(csum "Training complete" "$E" "$O")
  OOM=$(csum "CUDA out of memory|OutOfMemoryError" "$E" "$O")
  ERR=$(grep -hcE "Traceback|RuntimeError|ValueError|NotImplementedError|AssertionError|IndexError|size mismatch" "$E" 2>/dev/null)
  # ignore the benign torch-elastic exit-barrier teardown traceback (fires AFTER
  # 'Training complete'); only treat tracebacks as a problem if no completion yet
  HG=$(grep -hcE "Watchdog caught|operation timed out|ProcessGroupNCCL.*[Tt]imed out" "$E" 2>/dev/null)
  MT=$(stat -c %Y "$E" 2>/dev/null || echo 0); IDLE=$((NOW-MT))
  echo "[m$i +${EL}s] STAB-FULL:$ST ${STEP:-_}/200 m=${MAT:-0} dn=${DN:-0} oom=${OOM:-0} err=${ERR:-0} hg=${HG:-0} idle=${IDLE}s" >> "$STATUS"
  [ "${OOM:-0}" -gt 0 ] 2>/dev/null && problem="OOM"
  { [ "${ERR:-0}" -gt 0 ] && [ "${DN:-0}" -eq 0 ]; } 2>/dev/null && problem="ERR"
  [ "${HG:-0}" -gt 0 ] 2>/dev/null && problem="HANG"
  if [ "$ST" = "RUNNING" ] && [ "$MT" -gt 0 ] && [ "$IDLE" -gt 1500 ] && [ -n "${STEP:-}" ]; then problem="STALL>25min(step=${STEP})"; fi
  [ -n "$problem" ] && { echo "STAB_FULL_PROBLEM $problem (step=${STEP:-_})" >> "$STATUS"; break; }
  if [ "$ST" = "GONE" ] && [ "$SEEN" -eq 1 ]; then
    if [ "${DN:-0}" -gt 0 ]; then echo "STAB_FULL_GREEN (Training complete @200)" >> "$STATUS"
    else echo "STAB_FULL_GONE_NO_COMPLETE (investigate)" >> "$STATUS"; fi
    break
  fi
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "STAB_FULL_HEARTBEAT el=${EL}s step=${STEP:-_}" >> "$STATUS"; break; }
  sleep 120
done
