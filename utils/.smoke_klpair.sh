#!/bin/bash
# Matched KL-init DMD smoke pair on an existing allocation (ALLOC env or 6027557).
#   d10 = KL init, TF teacher only          (control / proper d3 rerun)
#   d9  = KL init, TF + AR dual head        (mode teacher on)
# Both with dfake_gen_update_ratio=1 so the GENERATOR actually updates
# (the 5:1 default makes any <6-step smoke test the critic only).
cd /scratch/u6ex/as1748.u6ex/ARRWM
KL=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt
SM=/scratch/u6ex/as1748.u6ex/ARRWM/analysis/.dmd_smoke_manifest.pt
A=${ALLOC:-6027557}

ALLOC=$A TAG=d10 STEPS=2 CKPT_INT=2 DMD_MANIFEST=$SM ODE_CKPT=$KL \
  bash sbatch/iact_dmd_smoke.sh max_rides=10 dfake_gen_update_ratio=1
echo "=== d10 (KL, TF only) exit=$? ==="

ALLOC=$A TAG=d9 STEPS=2 CKPT_INT=2 DMD_MANIFEST=$SM ODE_CKPT=$KL \
  bash sbatch/iact_dmd_smoke.sh max_rides=10 dfake_gen_update_ratio=1 dmd_ar_head_weight=1.0
echo "=== d9 (KL, TF+AR) exit=$? ==="

P=""
for t in d10 d9; do
  CK=logs/dmd10k_iact_$t/dmd10k_iact_$t/phase1_step0000002.pt
  [ -f "$CK" ] && P="$P vdmd_${t}_s2:$CK"
done
[ -n "$P" ] && PROBE_PAIRS="$P" bash utils/.probe_dmd_ckpts.sh
echo KLPAIR-DONE
