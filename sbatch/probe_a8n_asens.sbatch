#!/bin/bash
#SBATCH --job-name=probe-a8n-asens
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# ACTION-PROTECTED serve remap (lit-review 2026-08-07): per-channel blend
# scaled by w_j from the measured same-context action fan, so channels that
# carry action are corrected less. Targets the 62->50% assignment tax of the
# blanket remap. Fixed lam 1.0 / 0.75 + adaptive auto:1, all x asens.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
export ODE_EMDREMAP_ASENS=$PWD/analysis/eval_final/flow_viz/action_sensitivity_pilot4_alldir8n2.npz
CKPT=logs/ode14e_pilot/run3_flip2_alldir8n/action_ode_step0000400.pt
for MODE in 1.0 0.75 auto:1; do
  TAG=$(echo $MODE | tr -d '.:')
  ODE_EMDREMAP=$MODE FR_RUN=pilot5_a8n_as${TAG} FR_CONFIG=configs/ar_eval_dmd_student.yaml \
    FR_CKPT=$CKPT FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || echo "PROBE-FAIL as$TAG"
  FDD_NSEEDS=2 FDD_PAIRS="pilot5_a8n_as${TAG}=14e8" FDD_FIGNAME=flow_pilot5_a8n_as${TAG}_scorecard \
    python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL as$TAG"
  MC_RUNS=pilot5_a8n_as${TAG} python utils/motion_check.py || true
  SD_RUNS=pilot4_alldir8n2:pilot5_a8n_as${TAG} python utils/stat_drift.py || true
done
echo "PROBE-A8N-ASENS done $(date)"
