#!/bin/bash
# Watch the action-distribution analysis (5319236) + the gen smoke (5319074).
# On completion, dump the ACTION DISTRIBUTION table / smoke result into status.
AID=5319278; ABASE=analyze-actions-v14d
SID=5319279; SBASE=gen-lmdb-v14d-smoke
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/analyze_actions_status.txt
: > "$STATUS"
ASEEN=0; SSEEN=0
START=$(date +%s); HEARTBEAT=2400; i=0
jstate(){ local s; s=$(squeue -j "$1" -h -o "%T" 2>/dev/null); [ -z "$s" ] && echo GONE || echo "$s"; }
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START))
  AST=$(jstate $AID); [ "$AST" != GONE ] && ASEEN=1
  SST=$(jstate $SID); [ "$SST" != GONE ] && SSEEN=1
  AERR=$(grep -hcE "Traceback|Error:|RuntimeError|KeyError" "logs/${ABASE}_${AID}.err" 2>/dev/null)
  SERR=$(grep -hcE "Traceback|Error:|RuntimeError|KeyError" "logs/${SBASE}_${SID}.err" 2>/dev/null)
  MP4=$(ls -1 eval/ode_F_smoke/*.mp4 2>/dev/null | wc -l)
  echo "[m$i +${EL}s] analyze:$AST(err=${AERR:-0}) smoke:$SST(err=${SERR:-0} mp4=${MP4})" >> "$STATUS"
  done_a=0; done_s=0
  [ "$AST" = GONE ] && [ "$ASEEN" -eq 1 ] && done_a=1
  [ "$SST" = GONE ] && [ "$SSEEN" -eq 1 ] && done_s=1
  if [ "$done_a" -eq 1 ] && [ "$done_s" -eq 1 ]; then
    echo "=== ANALYZE TABLE ===" >> "$STATUS"
    grep -hA 9 "ACTION DISTRIBUTION" "logs/${ABASE}_${AID}.out" 2>/dev/null >> "$STATUS" || echo "(no table found)" >> "$STATUS"
    echo "=== SMOKE: ${MP4} mp4 in eval/ode_F_smoke ===" >> "$STATUS"
    echo "SMALL_JOBS_DONE" >> "$STATUS"
    break
  fi
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "ANALYZE_HEARTBEAT el=${EL}s analyze:$AST smoke:$SST" >> "$STATUS"; break; }
  sleep 60
done
