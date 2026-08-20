#!/bin/bash
# Checkpoint visualization watcher — runs INSIDE a node-holder job.
#
# Scans the campaign logdirs for new action_ode_step*.pt checkpoints and, for
# each one, renders on the holder's own GPUs (touching NOTHING in the training
# jobs): (1) generated videos (motion_check mp4s, 1 seed x 8 directions),
# (2) the flow-tree ellipse panel vs the untrained baseline. Per-checkpoint
# outputs are tagged v100_<arm>_s<step> so the evolution is browsable.
#
# Arms checkpoint every 200 steps (SAVE_EVERY spooled in their sbatch — the
# every-100 ask is satisfiable without resubmission only at this cadence for
# the ARMS; the new viz smokes checkpoint every 100). Sequential processing:
# one probe ~7-10 min, arrivals ~2-3/h across 5 sources — keeps up.
#
# Launch (from the holder command file):  bash utils/ckpt_viz_watcher.sh <hours>
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing
HOURS=${1:-4}
END=$((SECONDS + HOURS*3600 - 900))     # leave 15 min slack before walltime
DONE=logs/.viz_done
mkdir -p "$DONE"
FV=analysis/eval_final/flow_viz
echo "[vizw] up $(date) for ${HOURS}h on $(hostname)"

ARMS="run3_flip2_rollklts run3_flip2_rollmse9 run3_flip2_rollkl9 run3_flip2_rollklaw run3_flip2_rollklvz run3_flip2_rollklrep2 run3_flip2_rollkltssmoke run3_flip2_rollklrep2smoke"
while [ $SECONDS -lt $END ]; do
  for arm in $ARMS; do
    d=logs/ode14e_pilot/$arm
    [ -d "$d" ] || continue
    for ck in "$d"/action_ode_step*.pt; do
      [ -f "$ck" ] || continue
      step=$(basename "$ck" .pt | grep -oE '[0-9]+$')
      # skip step-0 startup ckpts and already-done ones
      [ "$((10#$step))" -eq 0 ] && continue
      tag="v100_${arm#run3_flip2_}_s${step}"
      [ -f "$DONE/$tag" ] && continue
      # ckpt may still be mid-write by the trainer: require 60s of quiet
      age=$(( $(date +%s) - $(stat -c %Y "$ck") ))
      [ "$age" -lt 60 ] && continue
      echo "[vizw] $(date +%H:%M) rendering $tag"
      srun --overlap --jobid=$SLURM_JOB_ID -N1 -n1 --gpus=1 --cpus-per-task=16 bash -c "
        cd /scratch/u6ex/as1748.u6ex/ARRWM
        source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
        export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=\$PWD:\$PWD/action-forcing
        FR_RUN=$tag FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$ck \
          FR_RUNGS='1000,625,357.142857,208.333333' FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=1 \
          python utils/flow_record_ode_student.py &&
        MC_OUT=$FV/.motion_check MC_RUNS=$tag python utils/motion_check.py || true
        FE_RUNS=pilot_gt0:$tag FE_OUT=$FV/flow_tree_ellipse_${tag}.png \
          FE_TITLE='baseline vs $tag' python utils/flow_tree_ellipse.py || true
      " > logs/vizw_${tag}.log 2>&1 \
        && touch "$DONE/$tag" && echo "[vizw] done $tag" \
        || echo "[vizw] FAILED $tag (logs/vizw_${tag}.log)"
      [ $SECONDS -ge $END ] && break 2
    done
  done
  sleep 60
done
echo "[vizw] window closed $(date)"
