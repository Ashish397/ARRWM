#!/bin/bash
# 10K-CAMPAIGN checkpoint watcher — runs INSIDE its own 2-node sbatch (NOT a
# holder: plain srun job steps get distinct GPUs natively, no --overlap
# pileup like the long36s0600 OOM). Four per-arm workers run concurrently,
# each processing that arm's checkpoints (100..500) in order:
#   (1) 6-chunk 1-seed probe + videos     -> tag v10k_<arm>_s<step>
#   (2) 15-chunk LEFT-only long video 16fps -> tag long15_<arm>_s<step>
#   (3) teacher_match @70f appended (flock) to teacher_match_v10k_evolution.csv
# Then, per step with all 4 arms present: 6-chunk grid video + long15 grid.
#   bash utils/ckpt_viz_watcher10k.sh <hours>
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing
HOURS=${1:-14}
END=$((SECONDS + HOURS*3600 - 900))
DONE=logs/.viz10k_done
mkdir -p "$DONE"
FV=analysis/eval_final/flow_viz
EVO=$FV/teacher_match_v10k_evolution.csv
ARMS="roll10k rollkl10k rollmse910k rollkl910k rolles10k"
echo "[v10k] up $(date) for ${HOURS}h on $(hostname)"

process_ckpt() {          # $1=arm $2=ckpt $3=step(7digit)
  local arm=$1 ck=$2 step=$3 tag=v10k_${1}_s${3} ltag=long15_${1}_s${3}
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
    MC_OUT=$FV/.motion_check MC_RUNS=$tag python utils/motion_check.py || true
    TM_RUNS=$tag TM_MAXF=70 TM_OUT=$FV/.tm_$tag.csv python utils/teacher_match.py || true
  " > logs/v10k_${tag}.log 2>&1 || return 1
  if [ -f "$FV/.tm_$tag.csv" ]; then
    flock logs/.tm10k.lock bash -c \
      "[ -f '$EVO' ] || head -1 '$FV/.tm_$tag.csv' > '$EVO'; \
       tail -n +2 '$FV/.tm_$tag.csv' >> '$EVO'; rm -f '$FV/.tm_$tag.csv'"
  fi
  return 0
}

arm_worker() {            # $1=arm — sequential over its own checkpoints
  local arm=$1 d=logs/ode14e_pilot/run3_flip2_$1
  for s in 100 200 300 400 500; do
    local step=$(printf %07d $s) ck=$d/action_ode_step$(printf %07d $s).pt
    while [ ! -f "$ck" ]; do
      [ $SECONDS -ge $END ] && return
      # trainer failed and will not deliver more ckpts -> stop waiting
      [ -f "$d/.train_done" ] && [ ! -f "$ck" ] && sleep 120 && \
        [ ! -f "$ck" ] && return
      sleep 60
    done
    # ckpt may be mid-write: require 60s of quiet
    while [ $(( $(date +%s) - $(stat -c %Y "$ck") )) -lt 60 ]; do sleep 30; done
    [ -f "$DONE/v10k_${arm}_s${step}" ] && continue
    echo "[v10k] $(date +%H:%M) rendering ${arm}@${s}"
    process_ckpt "$arm" "$ck" "$step" \
      && touch "$DONE/v10k_${arm}_s${step}" && echo "[v10k] done ${arm}@${s}" \
      || echo "[v10k] FAILED ${arm}@${s} (logs/v10k_v10k_${arm}_s${step}.log)"
    [ $SECONDS -ge $END ] && return
  done
}

for arm in $ARMS; do arm_worker "$arm" & done
wait

# ---- grids: one 6-chunk grid + one long15 grid per step -------------------
GVV="teacher:,mse:roll10k,mse9:rollmse910k,kl:rollkl10k,kl9:rollkl910k,es:rolles10k"
for s in 100 200 300 400 500; do
  step=$(printf %07d $s)
  n=$(ls -d $FV/.motion_check/v10k_*_s${step} 2>/dev/null | wc -l)
  [ "$n" -eq 0 ] && continue
  GV_TAGPREF=v10k GV_VARIANTS="$GVV" GV_DIR=L GV_STEPS=$s GV_FPS=16 \
    python utils/make_grid_videos.py || echo "[v10k] grid s$s failed"
  GVL="teacher:,mse:roll10k_s${step},mse9:rollmse910k_s${step},kl:rollkl10k_s${step},kl9:rollkl910k_s${step},es:rolles10k_s${step}"
  GV_LONG=long15 GV_VARIANTS="$GVL" GV_DIR=L GV_FPS=16 \
    GV_OUT=$FV/gridvid_L_long15s${step#000}.mp4 \
    python utils/make_grid_videos.py || echo "[v10k] long15 grid s$s failed"
done
echo "[v10k] window closed $(date)"
