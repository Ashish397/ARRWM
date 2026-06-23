#!/bin/bash
ID=5320890; BASE=classify-pool-v14d
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/classify_pool_status.txt; : > "$STATUS"; SEEN=0
START=$(date +%s); i=0
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START))
  ST=$(squeue -j $ID -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST=GONE
  [ "$ST" != GONE ] && SEEN=1
  ERR=$(grep -hcE "Traceback|Error:|RuntimeError|OutOfMemory" logs/${BASE}_${ID}.err 2>/dev/null)
  echo "[m$i +${EL}s] $ST err=${ERR:-0}" >> "$STATUS"
  [ "${ERR:-0}" -gt 0 ] && { echo "CLASSIFY_ERR" >> "$STATUS"; grep -hE "Traceback|Error" logs/${BASE}_${ID}.err 2>/dev/null | head -5 >> "$STATUS"; break; }
  if [ "$ST" = GONE ] && [ "$SEEN" -eq 1 ]; then
    echo "=== DIRECTION DISTRIBUTION (new turn def) + AVAILABILITY ===" >> "$STATUS"
    grep -hA 12 "ACTION DISTRIBUTION" logs/${BASE}_${ID}.out >> "$STATUS" 2>/dev/null
    grep -hA 14 "Total classified windows" logs/${BASE}_${ID}.out >> "$STATUS" 2>/dev/null
    echo "CLASSIFY_DONE" >> "$STATUS"; break
  fi
  [ "$EL" -ge 3000 ] && { echo "CLASSIFY_HEARTBEAT $ST" >> "$STATUS"; break; }
  sleep 60
done
