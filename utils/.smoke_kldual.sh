cd /scratch/u6ex/as1748.u6ex/ARRWM
KL=/scratch/u6ex/as1748.u6ex/ARRWM/logs/ode14e_pilot/run3_flip2_rollkl10k/action_ode_step0000400.pt
ALLOC=6027557 TAG=d9 STEPS=2 CKPT_INT=2 \
  DMD_MANIFEST=/scratch/u6ex/as1748.u6ex/ARRWM/analysis/.dmd_smoke_manifest.pt \
  ODE_CKPT=$KL bash sbatch/iact_dmd_smoke.sh max_rides=10 \
    dfake_gen_update_ratio=1 dmd_ar_head_weight=1.0
CK=logs/dmd10k_iact_d9/dmd10k_iact_d9/phase1_step0000002.pt
[ -f "$CK" ] && PROBE_PAIRS="vdmd_kldual_s2:$CK" bash utils/.probe_dmd_ckpts.sh
echo KLDUAL-SMOKE-DONE
