#!/bin/bash
# Confirm all 3 phase-3 DMD runs load their ODE ckpt + 14D teacher and reach training.
declare -A J=( [fwd800]=5355972 [nofwd800]=5355973 [fwd1000]=5355974 )
declare -A N=( [fwd800]=c8-sw-dmd3-fwd-ode800 [nofwd800]=c8-sw-dmd3-nofwd-ode800 [fwd1000]=c8-sw-dmd3-fwd-ode1000 )
ERRRE='Traceback|RuntimeError|KeyError|CUDA out of memory|ChildFailedError|size mismatch|Error\(s\) in loading|NCCL.*(error|timeout)|AssertionError'
lg(){ echo "logs/${N[$1]}_${J[$1]}.err"; }
report(){ echo "=== $(date -u +%H:%M:%S) ==="; squeue -j ${J[fwd800]},${J[nofwd800]},${J[fwd1000]} -o "%.10i %.24j %.9T %.8M %R" 2>/dev/null; }
start=$(date +%s)
for i in $(seq 1 240); do
  okcount=0
  for k in fwd800 nofwd800 fwd1000; do
    f=$(lg $k)
    e=$(grep -hcE "$ERRRE" "$f" 2>/dev/null | head -1)
    [ "${e:-0}" -gt 0 ] 2>/dev/null && { report; echo "ERROR in $k (${J[$k]}):"; grep -hE "$ERRRE" "$f" 2>/dev/null | tail -5; exit 1; }
    st=$(squeue -j ${J[$k]} -h -o "%T" 2>/dev/null)
    if [ -z "$st" ]; then fin=$(sacct -j ${J[$k]} -n -o State 2>/dev/null|head -1|tr -d ' '); echo "$k left queue: $fin"; if echo "$fin"|grep -qE "FAILED|CANCELLED|TIMEOUT|NODE_FAIL"; then report; tail -8 "$f" 2>/dev/null; exit 1; fi; fi
    # healthy markers: loaded the ODE ckpt + reached a step
    odeload=$(grep -hcE "Loading generator from .*action_ode_distill_F_noCD" "$f" 2>/dev/null | head -1)
    step=$(grep -hoE "step[ =][0-9]+" "$f" 2>/dev/null | tail -1 | grep -oE "[0-9]+")
    [ "${odeload:-0}" -ge 1 ] 2>/dev/null && [ "${step:-0}" -ge 1 ] 2>/dev/null && okcount=$((okcount+1))
  done
  if [ "$okcount" -ge 3 ]; then report; echo "ALL 3 LOADED ODE CKPT + REACHED TRAINING"; for k in fwd800 nofwd800 fwd1000; do echo "-- $k --"; grep -hE "Loading generator from|Loading.*teacher|v14_teacher|step[ =][0-9]+" "$(lg $k)" 2>/dev/null | tail -3; done; exit 0; fi
  echo "[t+$((($(date +%s)-start)/60))m] loaded+training: $okcount/3"
  sleep 60
done
report; echo "window elapsed"
