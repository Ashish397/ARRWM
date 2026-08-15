#!/bin/bash
# Interactive smoke sweep: 3-step train + step-3 video probe for every arm
# recipe not yet validated on the post-fix code. Runs through a HOLDER
# allocation (srun --overlap) — GPU-idle while its batch script does CPU
# tier-2 work. Sequential; each arm ~14 min (train ~9 + probe ~5).
#   ALLOC=<holder jobid> bash utils/smoke_all_arms.sh
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
ALLOC=${ALLOC:?}
FV=analysis/eval_final/flow_viz
RESULTS=logs/smoke_sweep_results.txt
: > $RESULTS

# arm | extra config args | extra env (colon-sep VAR=1 list, "-" for none)
SWEEP=(
  "mse9|ode_curriculum_mode=all9|-"
  "kled|ode_loss_type=kl_local ode_edist_weight=0.005|-"
  "all9aw|ode_actw_enabled=true ode_curriculum_mode=all9|-"
  "aws|ode_actw_enabled=true ode_curriculum_mode=actsplit|-"
  "klawv9|ode_loss_type=kl_local ode_actw_enabled=true ode_varw_enabled=true ode_curriculum_mode=all9|-"
  "klawvr|ode_loss_type=kl_local ode_actw_enabled=true ode_varw_enabled=true ode_gexcl_weight=0.5 ode_gexcl_band=0.05 ode_curriculum_mode=actsplit|-"
  "klawvr9|ode_loss_type=kl_local ode_actw_enabled=true ode_varw_enabled=true ode_gexcl_weight=0.5 ode_gexcl_band=0.05 ode_curriculum_mode=all9|-"
  "vz|ode_vrfm=true ode_vrfm_beta=1e-3|ODE_VRFM=1"
  "klvz|ode_loss_type=kl_local ode_vrfm=true ode_vrfm_beta=1e-3|ODE_VRFM=1"
  "klrep|ode_loss_type=kl_local ode_grep_weight=0.1 ode_grep_components=mu|-"
  "arattr|ode_attractor_enabled=true ode_attractor_weight=0.1 ode_attractor_warmup=5 ode_attractor_freeze_k=5|-"
)

for row in "${SWEEP[@]}"; do
  arm="${row%%|*}"; rest="${row#*|}"; extra="${rest%%|*}"; envs="${rest##*|}"
  TAG="sw_${arm}"
  echo "[sweep] $(date +%H:%M) === $arm ==="
  if [ "$envs" != "-" ]; then export ${envs//:/ }; fi
  ALLOC=$ALLOC TAG=$TAG STEPS=3 bash sbatch/iact_roll_smoke.sh $extra
  rc=$?
  if [ "$envs" != "-" ]; then for kv in ${envs//:/ }; do unset "${kv%%=*}"; done; fi
  ck=$(ls -t logs/ode14e_pilot/iact_${TAG}/action_ode_step0000003.pt 2>/dev/null | head -1)
  if [ $rc -ne 0 ] || [ -z "$ck" ]; then
    echo "$arm TRAIN-FAIL rc=$rc (logs/iact_roll_${TAG}.log)" >> $RESULTS
    echo "[sweep] $arm TRAIN-FAIL"; continue
  fi
  # step-3 video probe (1 seed, 8 direction videos)
  srun --overlap --jobid=$ALLOC -N1 -n1 --gpus=1 --cpus-per-task=16 bash -c "
    cd /scratch/u6ex/as1748.u6ex/ARRWM
    source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
    export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=\$PWD:\$PWD/action-forcing
    FR_RUN=$TAG FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$ck \
      FR_RUNGS='1000,625,357.142857,208.333333' FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=1 \
      python utils/flow_record_ode_student.py &&
    MC_OUT=$FV/.motion_check MC_RUNS=$TAG python utils/motion_check.py
  " > logs/sweep_probe_${TAG}.log 2>&1
  nv=$(ls $FV/.motion_check/${TAG}/r08_*_s0.mp4 2>/dev/null | wc -l)
  if [ "$nv" -ge 8 ]; then
    echo "$arm PASS (3 steps + $nv videos: $FV/.motion_check/${TAG}/)" >> $RESULTS
    echo "[sweep] $arm PASS ($nv videos)"
  else
    echo "$arm PROBE-FAIL videos=$nv (logs/sweep_probe_${TAG}.log)" >> $RESULTS
    echo "[sweep] $arm PROBE-FAIL"
  fi
done
echo "[sweep] complete $(date)"; cat $RESULTS
