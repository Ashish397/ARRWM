#!/bin/bash
# Tier-2 checkpoint analytics — runs on a holder, CONSUMES the video watcher's
# finished recordings (logs/.viz_done markers) and produces, per checkpoint:
#   (1) the flow-tree AR-timeline VIDEO  -> flow_tree_<tag>_AR_timeline.mp4
#   (2) teacher_match at the fair 70f horizon, appended to one evolution CSV
#       -> analysis/eval_final/flow_viz/teacher_match_v100_evolution.csv
# Distinct outputs from the video watcher => no write races. Single consumer
# per tag via logs/.viz_tl_done markers.
#   bash utils/ckpt_match_tl_loop.sh <hours>
set -u
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export PYTHONPATH=$PWD:$PWD/action-forcing
HOURS=${1:-6}
END=$((SECONDS + HOURS*3600 - 600))
mkdir -p logs/.viz_tl_done
FV=analysis/eval_final/flow_viz
EVO=$FV/teacher_match_v100_evolution.csv
echo "[t2] up $(date) for ${HOURS}h on $(hostname)"
while [ $SECONDS -lt $END ]; do
  for m in logs/.viz_done/v100_*; do
    [ -f "$m" ] || continue
    tag=$(basename "$m")
    [ -f "logs/.viz_tl_done/$tag" ] && continue
    echo "[t2] $(date +%H:%M) $tag"
    TL_WHICH=$tag TL_CUSTOM="$tag=$tag" python utils/flow_tree_timelines.py \
      > logs/t2_tl_$tag.log 2>&1 || echo "[t2] TL FAILED $tag"
    TM_RUNS=$tag TM_MAXF=70 TM_OUT=$FV/.tm_$tag.csv python utils/teacher_match.py \
      > logs/t2_tm_$tag.log 2>&1 \
      && { [ -f "$EVO" ] || head -1 $FV/.tm_$tag.csv > "$EVO"; \
           tail -n +2 $FV/.tm_$tag.csv >> "$EVO"; rm -f $FV/.tm_$tag.csv; } \
      || echo "[t2] TM FAILED $tag"
    touch "logs/.viz_tl_done/$tag"
    echo "[t2] done $tag"
    [ $SECONDS -ge $END ] && break
  done
  sleep 90
done
echo "[t2] window closed $(date)"
