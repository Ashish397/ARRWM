#!/bin/bash
# Fire the KL-init 35-step collapse test the moment the hung MSE torchrun
# step (6038535.14) releases the holder's GPUs. Detached via setsid+nohup so
# background-task cleanup on the agent side cannot kill the arming loop.
cd /scratch/u6ex/as1748.u6ex/ARRWM
STAMP=logs/.c35kl_arm.log
echo "[arm] waiting for 6038535.14 to clear ($(date))" >> $STAMP
while squeue -s -j 6038535 -h -o '%i' 2>/dev/null | grep -qx '6038535.14'; do
  sleep 45
done
echo "[arm] step cleared $(date) -> launching c35kl" >> $STAMP
ALLOC=6038535 TAG=c35kl STEPS=35 CKPT_EVERY=5 SAMP=5 \
  ODE_CKPT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt \
  bash sbatch/iact_dmd10k_aligned.sh >> $STAMP 2>&1
echo "[arm] c35kl finished rc=$? $(date)" >> $STAMP
