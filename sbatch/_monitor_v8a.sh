#!/bin/bash
# Monitor job 5043369 (v8a) until it clears step 50, or fails.
# Prints a status line each poll; exits 0 on success, 1 on failure,
# 2 if still in progress when the poll budget runs out (caller relaunches).
JOB=5043369
NAME=af-p3-cb-flashtr-v8a
OUT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/${NAME}_${JOB}.out
ERR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/${NAME}_${JOB}.err
TARGET=50
POLL=120
# ~50 min budget per launch; caller relaunches if still queued.
MAX_ITERS=25

latest_step() {
  # Extract the highest step=N/ reached across out+err.
  grep -hoE 'step=[0-9]+/' "$OUT" "$ERR" 2>/dev/null \
    | grep -oE '[0-9]+' | sort -n | tail -1
}

for i in $(seq 1 $MAX_ITERS); do
  STATE=$(squeue -j $JOB -h -o "%T" 2>/dev/null)
  if [ -z "$STATE" ]; then
    # Job no longer queued/running. Did it reach the target?
    ST=$(latest_step)
    if [ -n "$ST" ] && [ "$ST" -ge "$TARGET" ]; then
      echo "SUCCESS: job ended; reached step $ST (>= $TARGET)."
      exit 0
    fi
    echo "FAILURE: job $JOB not in queue and last step=${ST:-none} (< $TARGET)."
    echo "---- tail err ----"; tail -n 40 "$ERR" 2>/dev/null
    echo "---- tail out ----"; tail -n 20 "$OUT" 2>/dev/null
    exit 1
  fi
  if [ "$STATE" = "RUNNING" ]; then
    ST=$(latest_step)
    # Early crash detection while running.
    if grep -qE 'out of memory|CUDA error|Traceback \(most recent' "$ERR" 2>/dev/null; then
      echo "ERROR-SIGNATURE while RUNNING at step=${ST:-none}:"
      grep -nE 'out of memory|CUDA error|Traceback \(most recent|RuntimeError' "$ERR" 2>/dev/null | tail -n 8
    fi
    if [ -n "$ST" ] && [ "$ST" -ge "$TARGET" ]; then
      echo "SUCCESS: RUNNING and reached step $ST (>= $TARGET)."
      # Surface the R1/R2 cadence warning/info line if present.
      grep -hE 'LADD R2 enabled|R1 and R2 CO-FIRE' "$ERR" "$OUT" 2>/dev/null | tail -2
      exit 0
    fi
    echo "[$(date +%H:%M:%S)] RUNNING step=${ST:-0}/$TARGET (poll $i/$MAX_ITERS)"
  else
    echo "[$(date +%H:%M:%S)] STATE=$STATE (poll $i/$MAX_ITERS)"
  fi
  sleep $POLL
done
ST=$(latest_step)
echo "INPROGRESS: state=$(squeue -j $JOB -h -o '%T' 2>/dev/null) step=${ST:-0}/$TARGET — relaunch monitor."
exit 2
