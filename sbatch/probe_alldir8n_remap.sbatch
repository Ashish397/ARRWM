#!/bin/bash
#SBATCH --job-name=probe-a8n-remap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Best-trainer x best-serve: emdremap quantile transport on the alldir8n
# checkpoint (8-dir + noop trained), lambda sweep 0.5 / 0.75 / 1.0.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CKPT=logs/ode14e_pilot/run3_flip2_alldir8n/action_ode_step0000400.pt
for LAM in 0.5 0.75 1.0; do
  TAG=$(echo $LAM | tr -d .)
  ODE_EMDREMAP=$LAM FR_RUN=pilot5_a8n_rm${TAG} FR_CONFIG=configs/ar_eval_dmd_student.yaml \
    FR_CKPT=$CKPT FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || echo "PROBE-FAIL rm$TAG"
  FDD_NSEEDS=2 FDD_PAIRS="pilot5_a8n_rm${TAG}=14e8" FDD_FIGNAME=flow_pilot5_a8n_rm${TAG}_scorecard \
    python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL rm$TAG"
  MC_RUNS=pilot5_a8n_rm${TAG} python utils/motion_check.py || true
  SD_RUNS=pilot4_alldir8n2:pilot5_a8n_rm${TAG} python utils/stat_drift.py || true
done
echo "PROBE-A8N-REMAP done $(date)"
