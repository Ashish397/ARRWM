#!/bin/bash
# Fan out the remaining 5 DMD arms ONLY after the canary (dmd10k-mse) has
# proved the batch path on real 8-node hardware. Canary-first pattern: the
# 12 iact smokes ran inside an existing allocation and therefore tested
# neither the batch environment (TMPDIR leak) nor 8-node scheduling.
# Usage: CANARY=<jobid> bash utils/.fanout_dmd_arms.sh
cd /scratch/u6ex/as1748.u6ex/ARRWM
CANARY=${CANARY:?set CANARY=<jobid>}
MSE=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000200.pt
KL=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt
COMMON="MAXSTEPS=1000,CKPT_EVERY=100"

# Wait for the canary to prove itself: RUNNING and past step 50 with no
# OOM / PermissionError.
for i in $(seq 1 720); do
  S=$(squeue -j $CANARY -h -o '%T' 2>/dev/null)
  if [ -z "$S" ]; then echo "[fanout] canary $CANARY left the queue — ABORT, inspect it"; exit 1; fi
  if [ "$S" = "RUNNING" ]; then
    E=logs/dmd10k-mse_${CANARY}.err
    if grep -aqE 'OutOfMemory|PermissionError' "$E" 2>/dev/null; then
      echo "[fanout] canary FAILED with OOM/Permission — ABORT"; exit 1
    fi
    ST=$(grep -aoE 'step=[0-9]+' "$E" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
    if [ -n "$ST" ] && [ "$ST" -ge 50 ]; then
      echo "[fanout] canary healthy at step $ST — fanning out $(date)"
      sbatch -J dmd10k-kl      --export=ALL,DARM=kl,ODE_CKPT=$KL,$COMMON                        sbatch/train_dmd10k_stat.sbatch
      sbatch -J dmd10k-msedual --export=ALL,DARM=msedual,ODE_CKPT=$MSE,$COMMON,AR_HEAD=1.0      sbatch/train_dmd10k_stat.sbatch
      sbatch -J dmd10k-kldual  --export=ALL,DARM=kldual,ODE_CKPT=$KL,$COMMON,AR_HEAD=1.0        sbatch/train_dmd10k_stat.sbatch
      sbatch -J dmd10k-msear   --export=ALL,DARM=msear,ODE_CKPT=$MSE,$COMMON,AR_HEAD=1.0,TF_HEAD=0.0 sbatch/train_dmd10k_stat.sbatch
      sbatch -J dmd10k-klar    --export=ALL,DARM=klar,ODE_CKPT=$KL,$COMMON,AR_HEAD=1.0,TF_HEAD=0.0   sbatch/train_dmd10k_stat.sbatch
      echo "[fanout] 5 arms submitted"
      # Launch the viz watcher HERE, not on a bare "an arm is RUNNING" test:
      # arms that die in their first minutes would otherwise start the watcher
      # into an empty campaign, burning its 11h window (and holding 2 nodes the
      # arms are competing for). Step-50 health is the real green light.
      WJ=$(sbatch --parsable sbatch/viz_watcherdmd.sbatch)
      echo "[fanout] viz watcher submitted: $WJ"
      exit 0
    fi
  fi
  sleep 60
done
echo "[fanout] timed out waiting for canary"
