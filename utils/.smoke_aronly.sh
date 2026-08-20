#!/bin/bash
# AR-ONLY smoke pair: dmd_tf_head_weight=0 leaves the AR (mode-seeking)
# head as the sole DMD signal. Validates the new flag on both inits.
#   d11 = MSE init, AR only | d12 = KL init, AR only
cd /scratch/u6ex/as1748.u6ex/ARRWM
MSE=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_roll10k/action_ode_step0000200.pt
KL=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt
SM=/scratch/u6ex/as1748.u6ex/ARRWM/analysis/.dmd_smoke_manifest.pt
A=${ALLOC:-6027557}

ALLOC=$A TAG=d11 STEPS=2 CKPT_INT=2 DMD_MANIFEST=$SM ODE_CKPT=$MSE \
  bash sbatch/iact_dmd_smoke.sh max_rides=10 dfake_gen_update_ratio=1 \
    dmd_ar_head_weight=1.0 dmd_tf_head_weight=0.0
echo "=== d11 (MSE, AR only) exit=$? ==="

ALLOC=$A TAG=d12 STEPS=2 CKPT_INT=2 DMD_MANIFEST=$SM ODE_CKPT=$KL \
  bash sbatch/iact_dmd_smoke.sh max_rides=10 dfake_gen_update_ratio=1 \
    dmd_ar_head_weight=1.0 dmd_tf_head_weight=0.0
echo "=== d12 (KL, AR only) exit=$? ==="
echo ARONLY-DONE
