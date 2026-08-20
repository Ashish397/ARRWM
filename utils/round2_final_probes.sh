#!/bin/bash
# Round-2 closing sweep: 2-seed probes on PEAK + FINAL checkpoints of every
# round-2 arm (their 16h walls killed the in-job probe tails), then one
# combined teacher_match table at the fair horizon.
#   ALLOC=<holder> bash utils/round2_final_probes.sh
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
ALLOC=${ALLOC:?}
FV=analysis/eval_final/flow_viz
# arm:peak_step (from the evolution table; final = latest ckpt on disk)
SPECS="rollmse9:0000200 rollkl9:0001000 rollklaw:0000400 rollklvz:0000400 rollklrep2:0000400 rollklts:0000400"
RUNS=""
for spec in $SPECS; do
  arm=${spec%%:*}; pk=${spec##*:}
  fin=$(ls logs/ode14e_pilot/run3_flip2_${arm}/action_ode_step0*.pt 2>/dev/null | sort | tail -1 | grep -oE '[0-9]{7}')
  for st in $pk $fin; do
    [ -z "$st" ] && continue
    ck=logs/ode14e_pilot/run3_flip2_${arm}/action_ode_step${st}.pt
    [ -f "$ck" ] || continue
    tag=r2final_${arm}_s${st}
    [ -d "$FV/.motion_check/$tag" ] && { RUNS="$RUNS:$tag"; continue; }
    ee=""
    [ "$arm" = "rollklts" ] && ee="export ODE_KLTS_CKPT=$PWD/$ck;"
    echo "[r2] $(date +%H:%M) probing $tag"
    srun --overlap --jobid=$ALLOC -N1 -n1 --gpus=1 --cpus-per-task=16 bash -c "
      cd /scratch/u6ex/as1748.u6ex/ARRWM
      source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
      export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=\$PWD:\$PWD/action-forcing
      $ee FR_RUN=$tag FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$ck \
        FR_RUNGS='1000,625,357.142857,208.333333' FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 FR_FPS=16 \
        python utils/flow_record_ode_student.py
    " > logs/r2probe_${tag}.log 2>&1 && { echo "[r2] done $tag"; RUNS="$RUNS:$tag"; } \
      || echo "[r2] FAILED $tag"
  done
done
srun --overlap --jobid=$ALLOC -N1 -n1 --cpus-per-task=16 bash -c "
  cd /scratch/u6ex/as1748.u6ex/ARRWM
  source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
  PYTHONPATH=\$PWD TM_RUNS=${RUNS#:}:pilot_gt0 TM_MAXF=70 \
    TM_OUT=$FV/teacher_match_round2_final.csv \
    python utils/teacher_match.py > logs/tm_round2_final.log 2>&1
  grep -aE 'r2final|pilot_gt0' logs/tm_round2_final.log"
echo "[r2] sweep complete $(date)"
