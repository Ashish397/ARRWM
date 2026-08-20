#!/bin/bash
# DMD-10K checkpoint watcher — runs INSIDE its own 2-node sbatch (plain srun
# steps, no --overlap pileup). Three per-arm workers, checkpoints every 25
# steps (the early ones are the point: catch a DMD pull-off-course in
# minutes). Per checkpoint:
#   (1) 6-chunk 1-seed probe + videos     -> tag vdmd_<arm>_s<step>
#   (2) 15-chunk LEFT-only long video     -> tag long15dmd_<arm>_s<step>
#   (3) teacher_match @70f appended (flock) to teacher_match_vdmd_evolution.csv
# DMD ckpts may carry a different key layout than the ODE trainer's; if the
# probe rejects one, the failure lands in logs/vdmd_*.log for a key-adapter
# fix rather than silently skipping.
#   bash utils/ckpt_viz_watcherdmd.sh <hours>
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing
HOURS=${1:-12}
END=$((SECONDS + HOURS*3600 - 900))
DONE=logs/.vizdmd_done
mkdir -p "$DONE"
FV=analysis/eval_final/flow_viz
EVO=$FV/teacher_match_vdmd_evolution.csv
ARMS="mse kl msedual kldual msear klar"
echo "[vdmd] up $(date) for ${HOURS}h on $(hostname)"

process_ckpt() {          # $1=arm $2=ckpt $3=step(7digit)
  local arm=$1 ck=$2 step=$3 tag=vdmd_${1}_s${3} ltag=long15dmd_${1}_s${3}
  srun -N1 -n1 --gpus=1 --cpus-per-task=16 --exact bash -c "
    cd /scratch/u6ex/as1748.u6ex/ARRWM
    source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
    export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=\$PWD:\$PWD/action-forcing
    FR_RUN=$tag FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$ck \
      FR_RUNGS='1000,625,357.142857,208.333333' FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=1 FR_FPS=16 \
      python utils/flow_record_ode_student.py &&
    FR_RUN=$ltag FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$ck \
      FR_RUNGS='1000,625,357.142857,208.333333' FR_VIDEO=1 FR_CHUNKS=15 FR_NSEEDS=1 \
      FR_FPS=16 FR_DIRS=L \
      python utils/flow_record_ode_student.py &&
    MC_OUT=$FV/.motion_check MC_RUNS=$tag python utils/motion_check.py &&
    TM_RUNS=$tag TM_MAXF=70 TM_OUT=$FV/.tm_$tag.csv python utils/teacher_match.py
  " > logs/vdmd_${tag}.log 2>&1 || return 1
  # NOTE (review 2026-08-17): the trailing '|| true's used to swallow every
  # failure, so srun exited 0 even when the probe crashed and the step was
  # marked DONE forever. Failures must propagate to the '|| return 1' above.
  if [ -f "$FV/.tm_$tag.csv" ]; then
    flock logs/.tmdmd.lock bash -c \
      "[ -f '$EVO' ] || head -1 '$FV/.tm_$tag.csv' > '$EVO'; \
       tail -n +2 '$FV/.tm_$tag.csv' >> '$EVO'; rm -f '$FV/.tm_$tag.csv'"
  fi
  return 0
}

arm_worker() {            # $1=arm — sequential over its own checkpoints
  local arm=$1 d=logs/dmd10k_$1
  for s in $(seq 100 100 1000); do
    local step=$(printf %07d $s)
    # trainer checkpoint naming may differ (checkpoint_model_*.pt etc.) —
    # match any file carrying the padded step number.
    local ck=""
    while [ -z "$ck" ]; do
      # trainer nests ckpts under log_dir/<run_name>/phase1_step<7digit>.pt
      ck=$(ls "$d"/*${step}*.pt "$d"/*/*${step}*.pt 2>/dev/null | head -1)
      [ -n "$ck" ] && break
      [ $SECONDS -ge $END ] && return
      sleep 45
    done
    while [ $(( $(date +%s) - $(stat -c %Y "$ck") )) -lt 60 ]; do sleep 30; done
    [ -f "$DONE/vdmd_${arm}_s${step}" ] && continue
    echo "[vdmd] $(date +%H:%M) rendering ${arm}@${s} ($ck)"
    process_ckpt "$arm" "$ck" "$step" \
      && touch "$DONE/vdmd_${arm}_s${step}" && echo "[vdmd] done ${arm}@${s}" \
      || echo "[vdmd] FAILED ${arm}@${s} (logs/vdmd_vdmd_${arm}_s${step}.log)"
    [ $SECONDS -ge $END ] && return
  done
}

for arm in $ARMS; do arm_worker "$arm" & done
wait
# grids per step across the three DMD arms + teacher + the ODE init
for s in $(seq 100 100 1000); do
  step=$(printf %07d $s)
  n=$(ls -d $FV/.motion_check/vdmd_*_s${step} 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && continue
  GV_TAGPREF=vdmd GV_VARIANTS="teacher:,dmd-mse:mse,dmd-kl:kl,dmd-msedual:msedual,dmd-kldual:kldual,dmd-msear:msear,dmd-klar:klar" \
    GV_DIR=L GV_STEPS=$s GV_FPS=16 python utils/make_grid_videos.py \
    || echo "[vdmd] grid s$s failed"
done
echo "[vdmd] window closed $(date)"
