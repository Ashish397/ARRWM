#!/bin/bash
#SBATCH --job-name=probe-mn
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Probe the moment-loss arm with the PLAIN restart serve (no locks): the
# test is whether the TRAINING now preserves moments at the root.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CKPT=logs/ode14e_pilot/run3_flip2_mn/action_ode_step0000250.pt
FR_RUN=pilot4_mn2 FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$CKPT \
  FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
  python utils/flow_record_ode_student.py || echo "PROBE-FAIL mn"
FDD_NSEEDS=2 FDD_PAIRS="pilot4_mn2=14e8" FDD_FIGNAME=flow_pilot4_mn2_scorecard \
  python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL mn"
MC_RUNS=pilot4_mn2 python utils/motion_check.py || true
SD_RUNS=pilot3_flip2nr:pilot4_mn2 python utils/stat_drift.py || true
SF_RUNS=pilot_gt0:pilot4_mn2 python utils/stat_forensics.py || true
echo "PROBE-NRMOM done $(date)"
