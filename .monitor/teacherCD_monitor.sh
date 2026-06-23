#!/bin/bash
# Watch the teacher-CD-only run (5352500) to its step-100 eval; flag errors/stalls.
TC=5352500; NO=5350713
TCLOG=logs/ode-F-teaCD_${TC}.err
ERRRE='Traceback|RuntimeError|ValueError|AssertionError|ChildFailedError|CUDA out of memory|size mismatch|NCCL.*(error|timeout)|Watchdog'
cnt(){ grep -hcE "$1" "$2" 2>/dev/null | head -1; }
laststep(){ grep -oE "step=[0-9]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+"; }
report(){ echo "=== $(date -u +%H:%M:%S) ==="; squeue -j $TC,$NO -o "%.10i %.12j %.9T %.8M %R" 2>/dev/null; }
start=$(date +%s)
for i in $(seq 1 200); do
  st=$(squeue -j $TC -h -o "%T" 2>/dev/null)
  fin=""; [ -z "$st" ] && fin=$(sacct -j $TC -n -o State 2>/dev/null | head -1 | tr -d ' ')
  e=$(cnt "$ERRRE" "$TCLOG"); s=$(laststep "$TCLOG")
  if [ "${e:-0}" -gt 0 ] 2>/dev/null; then report; echo "TEACHER-CD ERROR (e=$e):"; grep -hE "$ERRRE" "$TCLOG" 2>/dev/null | tail -5; exit 1; fi
  if echo "$fin" | grep -qE "FAILED|CANCELLED|TIMEOUT|NODE_FAIL|OUT_OF"; then report; echo "teaCD ENDED: $fin"; tail -6 "$TCLOG" 2>/dev/null; exit 1; fi
  if [ "$st" = "RUNNING" ] && [ -f "$TCLOG" ]; then age=$(( $(date +%s) - $(stat -c %Y "$TCLOG") )); [ $age -gt 1200 ] && { report; echo "teaCD STALL idle ${age}s"; exit 2; }; fi
  # reached step-100 eval?
  if grep -qE "madrid_chain step=100" "$TCLOG" 2>/dev/null; then report; echo "TEACHER-CD reached step-100 eval:"; grep -hE "step=100 |\[eval step=100\]|madrid_chain step=100" "$TCLOG" 2>/dev/null | tail -3; echo "--- noCD step-100 for comparison ---"; grep -hE "\[eval step=100\]|madrid_chain step=100" logs/ode-F-noCD_${NO}.err 2>/dev/null | tail -2; exit 0; fi
  echo "[t+$((($(date +%s)-start)/60))m] teaCD:${st:-$fin} step=${s:-?}/1000 err=$e | startup_smoke=$(cnt 'Startup smoke: OK' "$TCLOG")"
  sleep 60
done
report; echo "window elapsed"
