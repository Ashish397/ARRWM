#!/bin/bash
#SBATCH --job-name=probe-nrsolvmid
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CKPT=logs/ode14e_pilot/run3_flip2_nrsolvmid/action_ode_step0000400.pt
FR_RUN=pilot6_nrsolvmid FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$CKPT \
  FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
  python utils/flow_record_ode_student.py || echo "PROBE-FAIL nrsolvmid"
FDD_NSEEDS=2 FDD_PAIRS="pilot6_nrsolvmid=14e8" FDD_FIGNAME=flow_pilot6_nrsolvmid_scorecard \
  python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL nrsolvmid"
MC_RUNS=pilot6_nrsolvmid python utils/motion_check.py || true
SD_RUNS=pilot4_alldir8n2:pilot6_nrsolvmid python utils/stat_drift.py || true
SF_RUNS=pilot_gt0:pilot6_nrsolvmid python utils/stat_forensics.py || true
echo "PROBE-nrsolvmid done $(date)"
