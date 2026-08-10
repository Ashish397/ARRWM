#!/bin/bash
#SBATCH --job-name=probe-a8n-samp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# INFERENCE-SIDE SOLVER on the best model (alldir8n): replace the x0-restart
# re-noise between rungs with a deterministic 2nd-order flow step (midpoint /
# Heun), velocity derived as v=(x-x0hat)/sigma. Euler baseline already exists
# as pilot4_alldir8n2. 2 forwards per rung -> ~2x probe cost.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CKPT=logs/ode14e_pilot/run3_flip2_alldir8n/action_ode_step0000400.pt
for S in midpoint heun; do
  ODE_SAMPLER=$S FR_RUN=pilot6_a8n_${S} FR_CONFIG=configs/ar_eval_dmd_student.yaml \
    FR_CKPT=$CKPT FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || echo "PROBE-FAIL $S"
  FDD_NSEEDS=2 FDD_PAIRS="pilot6_a8n_${S}=14e8" FDD_FIGNAME=flow_pilot6_a8n_${S}_scorecard \
    python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL $S"
  SD_RUNS=pilot4_alldir8n2:pilot6_a8n_${S} python utils/stat_drift.py || true
done
# FIXED motion analysis: one motion_check invocation over ALL comparands,
# then the CSV-based table (replaces the broken log parsing).
MC_RUNS=pilot4_alldir8n2:pilot6_a8n_midpoint:pilot6_a8n_heun:pilot5_a8n_rm10:pilot5_a8n_rmauto2:pilot5_a8n_rm05 \
  python utils/motion_check.py || true
MT_BASE=pilot4_alldir8n2 python utils/motion_table.py || true
echo "PROBE-A8N-SAMPLER done $(date)"
