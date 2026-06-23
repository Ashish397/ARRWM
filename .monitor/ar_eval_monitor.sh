#!/bin/bash
# Watch the two AR-chain eval jobs. First student-through-eval_causal_AR run, so
# the main risk is a config-load / strict-state-dict crash at startup -> surface fast.
declare -A J
J[5296677]="AR-FWD|ar-chain-statfwd|statfwd"
J[5296678]="AR-LOWT|ar-chain-statlowt|statlowt"
cd /scratch/u6ex/as1748.u6ex/ARRWM
STATUS=/tmp/ar_eval_status.txt
: > "$STATUS"
declare -A SEEN  # a job must be observed alive (PENDING/RUNNING) once before GONE counts as terminal
START=$(date +%s); HEARTBEAT=3600; i=0
while true; do
  i=$((i+1)); NOW=$(date +%s); EL=$((NOW-START))
  line="[m$i +${EL}s]"; problem=""; alldone=1
  for id in 5296677 5296678; do
    IFS='|' read -r nm jn lab <<< "${J[$id]}"
    OUT="logs/${jn}_${id}.err"; OUT2="logs/${jn}_${id}.out"
    ST=$(squeue -j $id -h -o "%T" 2>/dev/null); [ -z "$ST" ] && ST="GONE"
    DONEC=$(grep -hc "Done " "$OUT2" "$OUT" 2>/dev/null | paste -sd+ | bc 2>/dev/null)
    DCITY=$(grep -hcE "^done ${lab}" "$OUT2" "$OUT" 2>/dev/null | paste -sd+ | bc 2>/dev/null)
    FCITY=$(grep -hcE "^FAIL ${lab}" "$OUT2" "$OUT" 2>/dev/null | paste -sd+ | bc 2>/dev/null)
    ROLL=$(grep -hc "ROLLOUT_METRICS" eval/ar_chain/${lab}_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null)
    ERR=$(grep -hcE "Traceback|RuntimeError|Error loading|size mismatch|Missing key|KeyError|AssertionError" eval/ar_chain/${lab}_*.log 2>/dev/null | paste -sd+ | bc 2>/dev/null)
    line="$line | $nm:$ST done_cities=${DCITY:-0}/4 fail=${FCITY:-0} rollout_metrics=${ROLL:-0} ckpt_err=${ERR:-0}"
    [ "$ST" != "GONE" ] && SEEN[$id]=1
    # only terminal if we've actually seen it alive once (avoids the just-submitted squeue race)
    { [ "$ST" != "GONE" ] || [ -z "${SEEN[$id]:-}" ]; } && alldone=0
    [ "${FCITY:-0}" -gt 0 ] 2>/dev/null && problem="AR_FAIL:$nm($id) ${FCITY}city"
    [ "${ERR:-0}" -gt 0 ] 2>/dev/null && problem="AR_CKPT_ERR:$nm($id)"
  done
  echo "$line" >> "$STATUS"
  [ -n "$problem" ] && { echo "AR_PROBLEM $problem" >> "$STATUS"; break; }
  [ "$alldone" -eq 1 ] && { echo "AR_ALL_DONE (both eval jobs terminal)" >> "$STATUS"; break; }
  [ "$EL" -ge "$HEARTBEAT" ] && { echo "AR_HEARTBEAT el=${EL}s" >> "$STATUS"; break; }
  sleep 120
done
