#!/bin/bash
cd /scratch/u6ex/as1748.u6ex/ARRWM
KL=logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt
while true; do
  for H in $(squeue -u "$USER" -h -o "%i %j %T" | awk '$2 ~ /^hold-2n/ && $3=="RUNNING"{print $1}'); do
    [ "$(squeue -s -j "$H" -h -o '%j' 2>/dev/null | grep -cv batch)" != "0" ] && continue
    echo "$(date +%H:%M:%S) launch roll99_ar on $H | watcher" >> logs/_dispatch.log
    HOLDER=$H PORTOFF=6120 RUNSTAMP=$(date +%H%M%S) DARM=roll99_ar STAT_ANCHOR=1.0 WAVELET=true \
      AR_HEAD=1.0 TF_HEAD=0.0 MAXSTEPS=99 CKPT_EVERY=100000 ODE_CKPT=$KL \
      setsid nohup bash sbatch/_roll_holder.sh > logs/exp_roll99_ar.err 2>&1 < /dev/null &
    exit 0
  done
  sleep 45
done
