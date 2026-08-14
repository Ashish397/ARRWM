#!/bin/bash
#SBATCH --job-name=probe-mserep2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# MID-TRAINING probe of the four curriculum arms from whatever checkpoint
# exists now (step 700 ~ epoch 1.1), while training continues. Produces
# recordings + decoded videos + the teacher-match metric, so the arms are
# visible before the 20h runs end. Checkpoints are read-only here.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
RUNS=""
for A in c10mserep2; do
  D=logs/ode14e_pilot/run3_flip2_$A
  CK=$(ls -t $D/action_ode_step*.pt 2>/dev/null | grep -v step0000000 | head -1)
  if [ -z "$CK" ]; then echo "SKIP $A (no checkpoint yet)"; continue; fi
  STEP=$(basename $CK | grep -oE '[0-9]{7}')
  R=pilot7_${A}_s${STEP}
  echo "=== $A from $(basename $CK) -> $R ==="
  FR_RUN=$R FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$CK \
    FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || { echo "PROBE-FAIL $A"; continue; }
  FDD_NSEEDS=2 FDD_PAIRS="${R}=14e8" FDD_FIGNAME=flow_${R}_scorecard \
    python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL $A"
  SD_RUNS=pilot4_alldir8n2:${R} python utils/stat_drift.py || true
  RUNS="$RUNS:$R"
done
RUNS=${RUNS#:}
[ -n "$RUNS" ] && MC_RUNS="TEACHER48:pilot4_alldir8n2:$RUNS" python utils/motion_check.py || true
[ -n "$RUNS" ] && MT_BASE=TEACHER48 python utils/motion_table.py || true
[ -n "$RUNS" ] && TM_RUNS="pilot_gt0:pilot4_alldir8n2:$RUNS" \
  TM_OUT=analysis/eval_final/flow_viz/teacher_match_c10mid.csv \
  python utils/teacher_match.py || true
echo "PROBE-C10MID done $(date)"
