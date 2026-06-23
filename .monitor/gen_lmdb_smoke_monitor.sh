#!/bin/bash
# Watch the gen-lmdb-v14d SMOKE (5319074): gen 5 clean ODE pairs (14d) + decode
# to eval/ode_F_smoke. Reports completion / errors / output files.
ID=5319391
BASE=gen-lmdb-v14d-smoke
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/gen_lmdb_smoke_status.txt
: > "$STATUS"
SEEN=0
START=$(date +%s); HEARTBEAT=2400; i=0
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START)); problem=""
  E="logs/${BASE}_${ID}.err"; O="logs/${BASE}_${ID}.out"
  ST=$(squeue -j $ID -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST="GONE"
  [ "$ST" != "GONE" ] && SEEN=1
  GEN=$(grep -hcE "GPU 0:.*done=|win/s" "$E" "$O" 2>/dev/null | paste -sd+ | bc 2>/dev/null)
  DEC=$(grep -hcE "\[decode\] .*mp4=OK" "$E" "$O" 2>/dev/null | paste -sd+ | bc 2>/dev/null)
  DN=$(grep -hcE "smoke done" "$E" "$O" 2>/dev/null | paste -sd+ | bc 2>/dev/null)
  MP4=$(ls -1 eval/ode_F_smoke/*.mp4 2>/dev/null | wc -l)
  OOM=$(grep -hcE "CUDA out of memory|OutOfMemoryError" "$E" 2>/dev/null)
  ERR=$(grep -hcE "Traceback|RuntimeError|ValueError|KeyError|AssertionError|FileNotFoundError|Error:" "$E" 2>/dev/null)
  MT=$(stat -c %Y "$E" 2>/dev/null || echo 0); IDLE=$((NOW-MT))
  echo "[m$i +${EL}s] $ST gen=${GEN:-0} dec=${DEC:-0} mp4=${MP4} dn=${DN:-0} oom=${OOM:-0} err=${ERR:-0} idle=${IDLE}s" >> "$STATUS"
  [ "${OOM:-0}" -gt 0 ] 2>/dev/null && problem="OOM"
  [ "${ERR:-0}" -gt 0 ] 2>/dev/null && problem="ERR"
  [ -n "$problem" ] && { echo "GEN_SMOKE_PROBLEM $problem" >> "$STATUS"; break; }
  if [ "$ST" = "GONE" ] && [ "$SEEN" -eq 1 ]; then
    if [ "${DN:-0}" -gt 0 ] && [ "${MP4}" -ge 1 ]; then
      echo "GEN_SMOKE_GREEN (${MP4} mp4 in eval/ode_F_smoke, no errors)" >> "$STATUS"
    else
      echo "GEN_SMOKE_GONE_NO_COMPLETE (dn=${DN:-0} mp4=${MP4} err=${ERR:-0})" >> "$STATUS"
    fi
    break
  fi
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "GEN_SMOKE_HEARTBEAT el=${EL}s $ST gen=${GEN:-0} mp4=${MP4}" >> "$STATUS"; break; }
  sleep 60
done
