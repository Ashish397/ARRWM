#!/bin/bash
#SBATCH --job-name=probe-a8n-eta
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Adaptive-remap ETA KNEE HUNT on alldir8n. auto:1/:2 gave drift 0.828 ->
# 0.866/0.884 at only -1%/-5% motion (vs fixed lam=1.0: 0.936 but -49%
# motion). Push eta higher to find where motion damage starts.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CKPT=logs/ode14e_pilot/run3_flip2_alldir8n/action_ode_step0000400.pt
for ETA in 3 4 6; do
  ODE_EMDREMAP=auto:$ETA FR_RUN=pilot5_a8n_rmauto${ETA} FR_CONFIG=configs/ar_eval_dmd_student.yaml \
    FR_CKPT=$CKPT FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || echo "PROBE-FAIL auto$ETA"
  FDD_NSEEDS=2 FDD_PAIRS="pilot5_a8n_rmauto${ETA}=14e8" FDD_FIGNAME=flow_pilot5_a8n_rmauto${ETA}_scorecard \
    python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL auto$ETA"
  MC_RUNS=pilot5_a8n_rmauto${ETA} python utils/motion_check.py || true
  SD_RUNS=pilot4_alldir8n2:pilot5_a8n_rmauto${ETA} python utils/stat_drift.py || true
done
echo "PROBE-A8N-ETA done $(date)"
