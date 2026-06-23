#!/bin/bash
# Full-run monitor for both ODE-F runs. Exits on: any error, stall (>20min idle
# while RUNNING), or both finished. Otherwise watches to completion (~6h).
CD=5350712; NOCD=5350713
CDLOG=logs/ode-F_${CD}.err
NOLOG=logs/ode-F-noCD_${NOCD}.err
ERRRE='Traceback|RuntimeError|ValueError|AssertionError|ChildFailedError|CUDA out of memory|size mismatch|NCCL.*(error|timeout)|Watchdog'
cnt(){ grep -hcE "$1" "$2" 2>/dev/null | head -1 || echo 0; }   # robust single-int
laststep(){ grep -oE "step=[0-9]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+"; }
report(){ echo "=== $(date -u +%H:%M:%S) ==="; squeue -j $CD,$NOCD -o "%.10i %.12j %.9T %.8M %R" 2>/dev/null; }
start=$(date +%s)
for i in $(seq 1 460); do   # ~7.5h cap (60s tick)
  st_cd=$(squeue -j $CD -h -o "%T" 2>/dev/null); st_no=$(squeue -j $NOCD -h -o "%T" 2>/dev/null)
  fin_cd=""; fin_no=""
  [ -z "$st_cd" ] && fin_cd=$(sacct -j $CD -n -o State 2>/dev/null | head -1 | tr -d ' ')
  [ -z "$st_no" ] && fin_no=$(sacct -j $NOCD -n -o State 2>/dev/null | head -1 | tr -d ' ')
  e_cd=$(cnt "$ERRRE" "$CDLOG"); e_no=$(cnt "$ERRRE" "$NOLOG")
  s_cd=$(laststep "$CDLOG"); s_no=$(laststep "$NOLOG")
  # error
  if [ "${e_cd:-0}" -gt 0 ] 2>/dev/null || [ "${e_no:-0}" -gt 0 ] 2>/dev/null; then report; echo "ERROR cd=$e_cd noCD=$e_no"; grep -hE "$ERRRE" "$CDLOG" "$NOLOG" 2>/dev/null | tail -5; exit 1; fi
  # bad terminal state
  if echo "$fin_cd $fin_no" | grep -qE "FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF"; then report; echo "ENDED BADLY cd=$fin_cd noCD=$fin_no"; tail -6 "$CDLOG" "$NOLOG" 2>/dev/null; exit 1; fi
  # stall
  for pair in "CD:$st_cd:$CDLOG" "NOCD:$st_no:$NOLOG"; do nm=${pair%%:*}; rest=${pair#*:}; stt=${rest%%:*}; lg=${rest#*:}; 
    if [ "$stt" = "RUNNING" ] && [ -f "$lg" ]; then age=$(( $(date +%s) - $(stat -c %Y "$lg") )); [ $age -gt 1200 ] && { report; echo "STALL $nm idle ${age}s while RUNNING (possible hang)"; exit 2; }; fi; done
  # both done well
  if [ -n "$fin_cd" ] && [ -n "$fin_no" ] && echo "$fin_cd $fin_no" | grep -qE "COMPLETED"; then report; echo "BOTH COMPLETED cd=$fin_cd noCD=$fin_no"; exit 0; fi
  echo "[t+$((($(date +%s)-start)/60))m] cd:${st_cd:-$fin_cd} step=${s_cd:-?}/1000 err=$e_cd | noCD:${st_no:-$fin_no} step=${s_no:-?}/1000 err=$e_no"
  sleep 60
done
report; echo "7.5h monitor window elapsed (still running)"
