#!/bin/bash
#SBATCH --job-name=probe-a8n-gx
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# AXIS 2 AT INFERENCE: bounded gaussian exclusion at commit (eta sweep) on
# alldir8n, judged by the teacher-match metric + videos (NOT drift alone --
# the drift-only verdict on the quantile remap was wrong).
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CKPT=logs/ode14e_pilot/run3_flip2_alldir8n/action_ode_step0000400.pt
for ETA in 0.25 0.5 1.0; do
  TAG=$(echo $ETA | tr -d .)
  ODE_GEXCL=$ETA FR_RUN=pilot6_a8n_gx${TAG} FR_CONFIG=configs/ar_eval_dmd_student.yaml \
    FR_CKPT=$CKPT FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || echo "PROBE-FAIL gx$TAG"
  FDD_NSEEDS=2 FDD_PAIRS="pilot6_a8n_gx${TAG}=14e8" FDD_FIGNAME=flow_pilot6_a8n_gx${TAG}_scorecard \
    python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL gx$TAG"
  SD_RUNS=pilot4_alldir8n2:pilot6_a8n_gx${TAG} python utils/stat_drift.py || true
done
MC_RUNS=pilot4_alldir8n2:pilot6_a8n_gx025:pilot6_a8n_gx05:pilot6_a8n_gx10 python utils/motion_check.py || true
MT_BASE=pilot4_alldir8n2 python utils/motion_table.py || true
REF=""; [ -d analysis/eval_final/flow_viz/.motion_check/teacher20_matched ] && REF=teacher20_matched
TM_RUNS=pilot4_alldir8n2:pilot6_a8n_gx025:pilot6_a8n_gx05:pilot6_a8n_gx10 TM_REF_RUN=$REF \
  TM_OUT=analysis/eval_final/flow_viz/teacher_match_gexcl.csv python utils/teacher_match.py || true
echo "PROBE-A8N-GEXCL done $(date)"
