# Record + score each trained pilot on the r08 flow probe. Runs anywhere
# with 1 GPU (interactive overlap or batch). Skips missing/unfinished pilots.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp CUDA_VISIBLE_DEVICES=0
PAIRS=""
for V in gt flip2 dir4 dir8 mixed; do
  VDIR=$V; [ "$V" = "gt" ] && VDIR=gt_cap
  CKPT=$(ls logs/ode14e_pilot/run_${VDIR}/action_ode_step*.pt 2>/dev/null | tail -1)
  [ -z "$CKPT" ] && { echo "PROBE-SKIP $V (no ckpt)"; continue; }
  echo "=== probe $V ($CKPT) ==="
  FR_RUN=pilot_${V} FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$CKPT \
    FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || { echo "PROBE-FAIL $V"; continue; }
  PAIRS="${PAIRS}pilot_${V}=14e8:"
done
PAIRS=${PAIRS%:}
[ -n "$PAIRS" ] && FDD_PAIRS="$PAIRS" FDD_FIGNAME=flow_pilot_scorecard \
  python utils/flow_diverge_dmd3.py
echo "PROBE-PILOTS done"
