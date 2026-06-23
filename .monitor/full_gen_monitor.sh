#!/bin/bash
# Watch the full 6000-window balanced v14d gen (5322360, 8 nodes). Reports
# progress (done/skipped/failed across ranks), completion, errors/OOM.
ID=5323218; BASE=gen-lmdb-v14d
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/full_gen_status.txt; : > "$STATUS"; SEEN=0
START=$(date +%s); HEARTBEAT=12000; SAW_RUN=0; i=0
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START)); problem=""
  E="logs/${BASE}_${ID}.err"; O="logs/${BASE}_${ID}.out"
  ST=$(squeue -j $ID -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST=GONE
  [ "$ST" != GONE ] && SEEN=1
  PT=$(ls -1 /projects/u6ex/fbots/frodobots_lmdb/v14d/*.pt 2>/dev/null | wc -l)
  DONE=$(grep -hoE "FINISHED: done=[0-9]+" "$E" "$O" 2>/dev/null | grep -oE "[0-9]+" | paste -sd+ | bc 2>/dev/null)
  OOM=$(grep -hcE "CUDA out of memory|OutOfMemoryError|oom_kill|Out Of Memory" "$E" 2>/dev/null)
  ERR=$(grep -hcE "Traceback|RuntimeError|ValueError|KeyError|AssertionError|FileNotFoundError" "$E" 2>/dev/null)
  echo "[m$i +${EL}s] $ST pt=${PT}/6000 finished_sum=${DONE:-0} oom=${OOM:-0} err=${ERR:-0}" >> "$STATUS"
  if [ "$ST" = RUNNING ] && [ "$SAW_RUN" -eq 0 ]; then SAW_RUN=1; echo "FULL_GEN_STARTED (running on 8 nodes)" >> "$STATUS"; fi
  [ "${OOM:-0}" -gt 0 ] 2>/dev/null && problem="OOM"
  [ "${ERR:-0}" -gt 0 ] 2>/dev/null && problem="ERR"
  [ -n "$problem" ] && { echo "FULL_GEN_PROBLEM $problem (pt=${PT})" >> "$STATUS"; break; }
  if [ "$ST" = GONE ] && [ "$SEEN" -eq 1 ]; then
    if [ "${PT}" -ge 5800 ] 2>/dev/null; then echo "FULL_GEN_GREEN (${PT}/6000 .pt written)" >> "$STATUS";
    else echo "FULL_GEN_INCOMPLETE (${PT}/6000 -> resume-safe, may need requeue)" >> "$STATUS"; fi
    break
  fi
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "FULL_GEN_HEARTBEAT $ST pt=${PT}/6000" >> "$STATUS"; break; }
  sleep 120
done
