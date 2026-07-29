#!/bin/bash
#SBATCH --job-name=probe2-gt
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Probe the pilots-v2 gt variant (fixed objective, run2_gt_cap@250): the
# in-job probe crashed on the leaked AF_SNAPSHOT_STEPS training env
# (recorder now sanitizes it, but the job's gt probe slot already passed).
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CKPT=logs/ode14e_pilot/run2_gt_cap/action_ode_step0000250.pt
FR_RUN=pilot2_gt FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$CKPT \
  FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
  python utils/flow_record_ode_student.py || echo "PROBE-FAIL pilot2_gt"
FDD_NSEEDS=2 FDD_PAIRS="pilot2_gt=14e8" FDD_FIGNAME=flow_pilot2_gt_scorecard \
  python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL pilot2_gt"
MC_RUNS=pilot2_gt python utils/motion_check.py || true
echo "PROBE2-GT done"
