#!/bin/bash
FULL=5363856; SMK=5363855
FLOG=logs/c8-sw-freal-pfnoise_${FULL}.err; SLOG=logs/c8-sw-freal-pfnoise-smk_${SMK}.err
ERR='Traceback|RuntimeError|CUDA out of memory|OutOfMemory|ChildFailedError|size mismatch|NCCL.*(error|timeout)|Watchdog|AssertionError|mark.*ready.*twice'
cnt(){ grep -hcE "$ERR" "$1" 2>/dev/null|head -1; }
step(){ grep -hoE "step[ =][0-9]+" "$1" 2>/dev/null|tail -1|grep -oE "[0-9]+"; }
rep(){ echo "=== $(date -u +%H:%M:%S) ==="; squeue -j $FULL,$SMK -o "%.10i %.26j %.9T %.8M %R" 2>/dev/null; }
start=$(date +%s)
for i in $(seq 1 220); do
  e=$(cnt "$FLOG"); [ "${e:-0}" -gt 0 ] 2>/dev/null && { rep; echo "FULL ERROR:"; grep -hE "$ERR" "$FLOG" 2>/dev/null|grep -viE "destroy_process_group"|tail -6; exit 1; }
  st=$(squeue -j $FULL -h -o "%T" 2>/dev/null)
  if [ -z "$st" ]; then fin=$(sacct -j $FULL -n -o State 2>/dev/null|head -1|tr -d ' '); rep; echo "FULL FINISHED: $fin"; echo "$fin"|grep -qE "FAILED|TIMEOUT|NODE_FAIL" && { tail -6 "$FLOG" 2>/dev/null; exit 1; }; exit 0; fi
  if [ "$st" = "RUNNING" ] && [ -f "$FLOG" ]; then age=$(( $(date +%s)-$(stat -c %Y "$FLOG") )); [ $age -gt 1500 ] && { rep; echo "FULL STALL idle ${age}s"; exit 2; }; fi
  echo "[t+$((($(date +%s)-start)/60))m] full=${st}/step=$(step "$FLOG")/200 | smoke=$(squeue -j $SMK -h -o %T 2>/dev/null)/step=$(step "$SLOG")"
  sleep 90
done
rep; echo "window elapsed"
