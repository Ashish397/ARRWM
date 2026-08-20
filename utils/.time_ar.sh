#!/bin/bash
# Wall-time comparison at the ARMS' real cadence (dfake_gen_update_ratio=5):
# 20 iterations = 4 generator updates. Answers "do the AR arms fit in 6h for
# 1000 iterations" without guessing from noisy 2-step smokes.
cd /scratch/u6ex/as1748.u6ex/ARRWM
MSE=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000200.pt
SM=/scratch/u6ex/as1748.u6ex/ARRWM/analysis/.dmd_smoke_manifest.pt
A=${ALLOC:-6027557}

for cfg in tf ar; do
  EXTRA=""
  [ "$cfg" = "ar" ] && EXTRA="dmd_ar_head_weight=1.0"
  T0=$(date +%s)
  ALLOC=$A TAG=t$cfg STEPS=20 CKPT_INT=100000 DMD_MANIFEST=$SM ODE_CKPT=$MSE \
    bash sbatch/iact_dmd_smoke.sh max_rides=10 $EXTRA >/dev/null 2>&1
  T1=$(date +%s)
  # subtract startup by reading the trainer's own first/last step timestamps
  L=logs/iact_dmd_t$cfg.log
  S=$(grep -a 'Trainer ready' $L | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
  E=$(grep -a 'Training complete' $L | grep -oE '[0-9]{2}:[0-9]{2}:[0-9]{2}' | head -1)
  echo "cfg=$cfg total=$((T1-T0))s trainer_window=${S}..${E}"
done
echo TIMING-DONE
