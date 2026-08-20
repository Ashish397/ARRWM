#!/bin/bash
# File-driven experiment dispatcher. Reads sbatch/_exp_queue.txt every loop,
# skips any tag already recorded in logs/_dispatch.log, and launches the next
# pending arm onto any hold-2n* holder that is RUNNING with no active step.
# Safe to edit the queue file or restart this script at any time: launched
# tags are remembered via the log, so nothing is ever run twice.
cd /scratch/u6ex/as1748.u6ex/ARRWM
MSE=logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000200.pt
KL=logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt
touch logs/_dispatch.log
while true; do
  PENDING=0
  while IFS='|' read -r TAG LAUNCH EXTRA CKPTK ARH WAV; do
    [ -z "$TAG" ] && continue
    case "$TAG" in \#*) continue;; esac
    grep -q "launch $TAG " logs/_dispatch.log && continue
    PENDING=$((PENDING+1))
    for H in $(squeue -u "$USER" -h -o "%i %j %T" | awk '$2 ~ /^hold-2n/ && $3=="RUNNING"{print $1}'); do
      [ "$(squeue -s -j "$H" -h -o '%j' 2>/dev/null | grep -cv batch)" != "0" ] && continue
      [ "$CKPTK" = "KL" ] && CK=$KL || CK=$MSE
      export HOLDER=$H PORTOFF=$(( 1000 + (RANDOM % 8000) )) RUNSTAMP=$(date +%H%M%S)
      case "$TAG" in sa_w01) SA=0.1;; *) SA=1.0;; esac
    export DARM="$TAG" STAT_ANCHOR=$SA MAXSTEPS=91 CKPT_EVERY=100000
      export ODE_CKPT=$CK DEXTRA="$EXTRA" WAVELET=$WAV
      if [ -n "$ARH" ]; then export AR_HEAD=$ARH TF_HEAD=0.0; else unset AR_HEAD TF_HEAD; fi
      echo "$(date +%H:%M:%S) launch $TAG on $H | wav=$WAV | $EXTRA" >> logs/_dispatch.log
      setsid nohup bash "sbatch/$LAUNCH" > "logs/exp_${TAG}.err" 2>&1 < /dev/null &
      sleep 120
      break
    done
  done < sbatch/_exp_queue.txt
  [ "$PENDING" = "0" ] && { echo "$(date +%H:%M:%S) ALL EXPERIMENTS DISPATCHED" >> logs/_dispatch.log; break; }
  sleep 60
done
