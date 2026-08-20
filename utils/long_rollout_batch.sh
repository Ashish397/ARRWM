#!/bin/bash
# LONG AUTOREGRESSIVE ROLLOUTS: 36 chunks (~110 s of video) at 16 FPS for the
# grid variants, latest checkpoint each, on a holder's GPUs. 2 concurrent
# probes (one per node) to share with the viz stack.
#   ALLOC=<holder jobid> bash utils/long_rollout_batch.sh
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
ALLOC=${ALLOC:?}
FV=analysis/eval_final/flow_viz
latest() { ls logs/ode14e_pilot/run3_flip2_$1/action_ode_step0*.pt 2>/dev/null | sort | tail -1; }
run_one() {
  local var=$1; local ck=$2; local extra_env=$3
  local tag=long36_${var}
  srun --overlap --jobid=$ALLOC -N1 -n1 --gpus=1 --cpus-per-task=16 bash -c "
    cd /scratch/u6ex/as1748.u6ex/ARRWM
    source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
    export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=\$PWD:\$PWD/action-forcing
    $extra_env
    FR_RUN=$tag FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$ck \
      FR_RUNGS='1000,625,357.142857,208.333333' FR_VIDEO=1 FR_CHUNKS=36 \
      FR_NSEEDS=1 FR_FPS=16 \
      python utils/flow_record_ode_student.py
  " > logs/long36_${var}.log 2>&1 \
    && echo "[long] done $var ($(ls $FV/.motion_check/$tag/*.mp4 2>/dev/null | wc -l) videos)" \
    || echo "[long] FAILED $var (logs/long36_${var}.log)"
}
echo "[long] start $(date)"
i=0
for var in roll rollkl rollkl9 rollklvz rollklaw rollklrep2 rollklts; do
  ck=$(latest $var)
  [ -z "$ck" ] && { echo "[long] no ckpt for $var"; continue; }
  ee=""
  [ "$var" = "rollklts" ] && ee="export ODE_KLTS_CKPT=\$PWD/$ck"
  run_one "$var" "$ck" "$ee" &
  i=$((i+1))
  [ $((i % 2)) -eq 0 ] && wait
done
wait
echo "[long] all done $(date)"
