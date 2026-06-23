#!/bin/bash
# Wake at a milestone: report when BOTH CD runs have a Madrid eval at >= TARGET,
# or earlier on error/collapse/job-end. Args: $1=TARGET step (default 300).
TARGET=${1:-300}
BOTHCD=5353964; TEACD=5352500; NOCD=5350713
BLOG=logs/ode-F_${BOTHCD}.err; TLOG=logs/ode-F-teaCD_${TEACD}.err; NLOG=logs/ode-F-noCD_${NOCD}.err
ERRRE='Traceback|RuntimeError|CUDA out of memory|ChildFailedError|size mismatch|NCCL.*(error|timeout)|Watchdog'
mad(){ grep -hE "madrid_chain step=" "$1" 2>/dev/null | grep -oE "step=[0-9]+.*musiq=[0-9.]+ niqe=[0-9.]+ late-early=[-0-9.]+"; }
madstep(){ grep -hoE "madrid_chain step=[0-9]+" "$1" 2>/dev/null | grep -oE "[0-9]+" | tail -1; }
curstep(){ grep -hoE "step=[0-9]+ " "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+"; }
emit(){ echo "######## MILESTONE $TARGET ($(date -u +%H:%M:%S)) ########";
  for p in "both-CD:$BLOG" "teaCD:$TLOG" "noCD(baseline):$NLOG"; do echo "== ${p%%:*} =="; mad "${p#*:}" | tail -7; done; }
start=$(date +%s)
for i in $(seq 1 300); do
  for p in "both-CD:$BOTHCD:$BLOG" "teaCD:$TEACD:$TLOG"; do nm=${p%%:*}; rest=${p#*:}; jid=${rest%%:*}; lg=${rest#*:}
    e=$(grep -hcE "$ERRRE" "$lg" 2>/dev/null | head -1)
    [ "${e:-0}" -gt 0 ] 2>/dev/null && { echo "ERROR in $nm ($jid):"; grep -hE "$ERRRE" "$lg" 2>/dev/null | tail -4; emit; exit 1; }
    st=$(squeue -j $jid -h -o "%T" 2>/dev/null); [ -z "$st" ] && { fin=$(sacct -j $jid -n -o State 2>/dev/null|head -1|tr -d ' '); echo "$nm ($jid) left queue: $fin"; emit; exit 3; }
    if [ "$st" = "RUNNING" ] && [ -f "$lg" ]; then age=$(( $(date +%s) - $(stat -c %Y "$lg") )); [ $age -gt 1200 ] && { echo "STALL $nm idle ${age}s"; emit; exit 2; }; fi
  done
  bms=$(madstep "$BLOG"); tms=$(madstep "$TLOG")
  if [ "${bms:-0}" -ge "$TARGET" ] && [ "${tms:-0}" -ge "$TARGET" ]; then emit; echo "BOTH CD RUNS REACHED MILESTONE $TARGET"; exit 0; fi
  echo "[t+$((($(date +%s)-start)/60))m] both-CD step=$(curstep "$BLOG")/madrid=$bms | teaCD step=$(curstep "$TLOG")/madrid=$tms | target=$TARGET"
  sleep 60
done
emit; echo "window elapsed"
