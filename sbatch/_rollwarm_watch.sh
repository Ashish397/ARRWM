#!/bin/bash
# Fires the two warm-start rolling arms on the next two free holders.
cd /scratch/u6ex/as1748.u6ex/ARRWM
KL=logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt
MSE=logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000200.pt
QUEUE=(
 "rollwarm_kl|logs/dmd10k_dmd3kl/dmd10k_dmd3kl_j6052054/phase1_step0000200.pt|$KL"
 "rollwarm_mse|logs/dmd10k_dmd3mse/dmd10k_dmd3mse_j6052053/phase1_step0000200.pt|$MSE"
)
i=0
while [ $i -lt ${#QUEUE[@]} ]; do
  for H in $(squeue -u "$USER" -h -o "%i %j %T" | awk '$2 ~ /^hold-2n/ && $3=="RUNNING"{print $1}'); do
    [ $i -ge ${#QUEUE[@]} ] && break
    [ "$(squeue -s -j "$H" -h -o '%j' 2>/dev/null | grep -cv batch)" != "0" ] && continue
    L=$(squeue -j "$H" -h -o %L); case "$L" in *:*:*) ;; *) continue;; esac
    IFS='|' read -r TAG SRC OCK <<< "${QUEUE[$i]}"
    echo "$(date +%H:%M:%S) rollwarm $TAG on $H" >> logs/_dispatch.log
    HOLDER=$H PORTOFF=$((3000+i*457)) DARM=$TAG SRC=$SRC ODE_CKPT=$OCK \
      setsid nohup bash sbatch/_rollwarm.sh > "logs/exp_${TAG}.err" 2>&1 < /dev/null &
    i=$((i+1)); sleep 150; break
  done
  sleep 60
done
echo "$(date +%H:%M:%S) rollwarm both dispatched" >> logs/_dispatch.log
