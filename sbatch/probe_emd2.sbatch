#!/bin/bash
#SBATCH --job-name=probe-emd2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Forensics probe for the dir8n emd2 arm (500-step checkpoint).
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CKPT=logs/ode14e_pilot/run3_flip2_emd2/action_ode_step0000500.pt
FR_RUN=pilot4_emd22 FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$CKPT \
  FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
  python utils/flow_record_ode_student.py || echo "PROBE-FAIL emd2"
FDD_NSEEDS=2 FDD_PAIRS="pilot4_emd22=14e8" FDD_FIGNAME=flow_pilot4_emd22_scorecard \
  python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL emd2"
MC_RUNS=pilot4_emd22 python utils/motion_check.py || true
SD_RUNS=pilot2_flip2:pilot4_emd22 python utils/stat_drift.py || true
SF_RUNS=pilot_gt0:pilot4_emd22 python utils/stat_forensics.py || true
echo "PROBE-emd2 done $(date)"
