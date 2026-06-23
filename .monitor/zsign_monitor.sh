#!/bin/bash
ID=5319957; BASE=analyze-actions-v14d
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/zsign_status.txt; : > "$STATUS"; SEEN=0
START=$(date +%s); i=0
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START))
  ST=$(squeue -j $ID -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST=GONE
  [ "$ST" != GONE ] && SEEN=1
  echo "[m$i +${EL}s] $ST" >> "$STATUS"
  if [ "$ST" = GONE ] && [ "$SEEN" -eq 1 ]; then
    echo "=== REPORT ===" >> "$STATUS"
    grep -hA 20 "ACTION DISTRIBUTION" logs/${BASE}_${ID}.out >> "$STATUS" 2>/dev/null
    echo "ZSIGN_DONE" >> "$STATUS"; break
  fi
  [ "$EL" -ge 3000 ] && { echo "ZSIGN_HEARTBEAT" >> "$STATUS"; break; }
  sleep 45
done
