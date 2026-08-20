cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing
for pair in $PROBE_PAIRS; do
  tag=${pair%%:*}; ck=${pair##*:}
  [ -d analysis/eval_final/flow_viz/.motion_check/$tag ] && { echo "skip $tag"; continue; }
  FR_RUN=$tag FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$ck \
    FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=1 FR_FPS=16 \
    python utils/flow_record_ode_student.py || echo "FAIL $tag"
done
TM_RUNS=$(echo $PROBE_PAIRS | tr ' ' '\n' | cut -d: -f1 | paste -sd:) TM_MAXF=70 \
  TM_OUT=analysis/eval_final/flow_viz/.tm_dmd2.csv python utils/teacher_match.py
echo PROBES2-DONE
