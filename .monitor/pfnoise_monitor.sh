#!/bin/bash
# Watch the pfnoise smoke (5363855) + full (5363856). Flag errors/OOM/hang;
# report when the SMOKE validates the new de-drift/latent path (reaches step>=25).
SMK=5363855; FULL=5363856
SLOG=logs/c8-sw-freal-pfnoise-smk_${SMK}.err; FLOG=logs/c8-sw-freal-pfnoise_${FULL}.err
ERR='Traceback|RuntimeError|KeyError|CUDA out of memory|OutOfMemory|ChildFailedError|size mismatch|NCCL.*(error|timeout)|Watchdog|AssertionError|prepare_for_backward|mark.*ready.*twice'
cnt(){ grep -hcE "$ERR" "$1" 2>/dev/null | head -1; }
step(){ grep -hoE "step[ =][0-9]+" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+"; }
rep(){ echo "=== $(date -u +%H:%M:%S) ==="; squeue -j $SMK,$FULL -o "%.10i %.26j %.9T %.8M %R" 2>/dev/null; }
start=$(date +%s)
for i in $(seq 1 240); do
  for p in "smoke:$SMK:$SLOG" "full:$FULL:$FLOG"; do nm=${p%%:*}; r=${p#*:}; j=${r%%:*}; lg=${r#*:}
    e=$(cnt "$lg"); [ "${e:-0}" -gt 0 ] 2>/dev/null && { rep; echo "ERROR in $nm ($j):"; grep -hE "$ERR" "$lg" 2>/dev/null|grep -viE "destroy_process_group"|tail -6; exit 1; }
    st=$(squeue -j $j -h -o "%T" 2>/dev/null)
    if [ -z "$st" ]; then fin=$(sacct -j $j -n -o State 2>/dev/null|head -1|tr -d ' '); echo "$nm ($j) left queue: $fin"; echo "$fin"|grep -qE "FAILED|TIMEOUT|NODE_FAIL" && { rep; tail -8 "$lg" 2>/dev/null; exit 1; }; fi
    if [ "$st" = "RUNNING" ] && [ -f "$lg" ]; then age=$(( $(date +%s)-$(stat -c %Y "$lg") )); [ $age -gt 1500 ] && { rep; echo "STALL $nm idle ${age}s"; exit 2; }; fi
  done
  ssm=$(step "$SLOG"); 
  if [ "${ssm:-0}" -ge 25 ] 2>/dev/null; then rep; echo "SMOKE VALIDATED de-drift+latent path (step=$ssm):"; grep -hE "Loading generator from|fn_fwd_loss|fn_rev_loss|fn_cyc_loss|step=25 |madrid|dmd_mae_gate_m_fake" "$SLOG" 2>/dev/null|tail -4; exit 0; fi
  echo "[t+$((($(date +%s)-start)/60))m] smoke=${st:-_}/step=${ssm:-_} | full=$(squeue -j $FULL -h -o %T 2>/dev/null)/step=$(step "$FLOG")"
  sleep 60
done
rep; echo "window elapsed"
