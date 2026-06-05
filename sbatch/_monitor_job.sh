#!/bin/bash
# Generic: monitor $1=jobid $2=jobname until it clears step 50, or fails.
# exit 0 = reached step 50; 1 = failed/ended before 50; 2 = still in progress.
JOB=$1
NAME=$2
OUT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/${NAME}_${JOB}.out
ERR=/scratch/u6ex/as1748.u6ex/ARRWM/logs/${NAME}_${JOB}.err
TARGET=50
POLL=120
MAX_ITERS=25

latest_step() {
  grep -hoE 'step=[0-9]+/' "$OUT" "$ERR" 2>/dev/null \
    | grep -oE '[0-9]+' | sort -n | tail -1
}

for i in $(seq 1 $MAX_ITERS); do
  STATE=$(squeue -j $JOB -h -o "%T" 2>/dev/null)
  if [ -z "$STATE" ]; then
    ST=$(latest_step)
    if [ -n "$ST" ] && [ "$ST" -ge "$TARGET" ]; then
      echo "SUCCESS($NAME/$JOB): job ended; reached step $ST (>= $TARGET)."
      exit 0
    fi
    echo "FAILURE($NAME/$JOB): not in queue, last step=${ST:-none} (< $TARGET)."
    echo "---- tail err ----"; tail -n 45 "$ERR" 2>/dev/null
    exit 1
  fi
  if [ "$STATE" = "RUNNING" ]; then
    ST=$(latest_step)
    if grep -qE 'out of memory|CUDA error|Traceback \(most recent|AssertionError|ValueError|RuntimeError' "$ERR" 2>/dev/null; then
      echo "ERROR-SIGNATURE($NAME) at step=${ST:-none}:"
      grep -nE 'out of memory|CUDA error|Traceback \(most recent|AssertionError|ValueError|RuntimeError' "$ERR" 2>/dev/null | tail -n 8
    fi
    if [ -n "$ST" ] && [ "$ST" -ge "$TARGET" ]; then
      echo "SUCCESS($NAME/$JOB): RUNNING and reached step $ST (>= $TARGET)."
      grep -hE 'LADD R2 enabled|R1 and R2 CO-FIRE|dmd_mae_gate' "$ERR" "$OUT" 2>/dev/null | tail -3
      exit 0
    fi
    echo "[$(date +%H:%M:%S)] $NAME RUNNING step=${ST:-0}/$TARGET (poll $i/$MAX_ITERS)"
  else
    echo "[$(date +%H:%M:%S)] $NAME STATE=$STATE (poll $i/$MAX_ITERS)"
  fi
  sleep $POLL
done
echo "INPROGRESS($NAME/$JOB): state=$(squeue -j $JOB -h -o '%T' 2>/dev/null) step=$(latest_step)/$TARGET — relaunch."
exit 2
