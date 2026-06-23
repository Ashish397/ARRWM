#!/bin/bash
ID=5322241; BASE=gen-lmdb-v14d-smoke
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/mixed_smoke_status.txt; : > "$STATUS"; SEEN=0
START=$(date +%s); i=0
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START))
  ST=$(squeue -j $ID -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST=GONE
  [ "$ST" != GONE ] && SEEN=1
  MP4=$(ls -1 eval/ode_F_smoke_mixed/*.mp4 2>/dev/null | wc -l)
  ERR=$(grep -hcE "Traceback|Error:|RuntimeError|OutOfMemory|KeyError" logs/${BASE}_${ID}.err 2>/dev/null)
  echo "[m$i +${EL}s] $ST mp4=${MP4} err=${ERR:-0}" >> "$STATUS"
  [ "${ERR:-0}" -gt 0 ] && { echo "MIXED_SMOKE_ERR" >> "$STATUS"; grep -hE "Error|Traceback" logs/${BASE}_${ID}.err 2>/dev/null|tail -4 >>"$STATUS"; break; }
  if [ "$ST" = GONE ] && [ "$SEEN" -eq 1 ]; then
    [ "$MP4" -ge 1 ] && echo "MIXED_SMOKE_GREEN ($MP4 mp4)" >> "$STATUS" || echo "MIXED_SMOKE_NO_MP4" >> "$STATUS"
    break
  fi
  [ "$EL" -ge 2400 ] && { echo "MIXED_SMOKE_HEARTBEAT $ST mp4=$MP4" >> "$STATUS"; break; }
  sleep 60
done
