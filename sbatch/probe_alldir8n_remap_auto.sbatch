#!/bin/bash
#SBATCH --job-name=probe-a8n-auto
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Adaptive emdremap (per-channel lambda from contraction deficit) on the
# alldir8n checkpoint, eta 1 and 2 — plus NOOP serve probes (does the
# remap inject phantom motion when holding still?).
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CKPT=logs/ode14e_pilot/run3_flip2_alldir8n/action_ode_step0000400.pt
for ETA in 1 2; do
  ODE_EMDREMAP=auto:$ETA FR_RUN=pilot5_a8n_rmauto${ETA} FR_CONFIG=configs/ar_eval_dmd_student.yaml \
    FR_CKPT=$CKPT FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || echo "PROBE-FAIL auto$ETA"
  FDD_NSEEDS=2 FDD_PAIRS="pilot5_a8n_rmauto${ETA}=14e8" FDD_FIGNAME=flow_pilot5_a8n_rmauto${ETA}_scorecard \
    python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL auto$ETA"
  MC_RUNS=pilot5_a8n_rmauto${ETA} python utils/motion_check.py || true
  SD_RUNS=pilot4_alldir8n2:pilot5_a8n_rmauto${ETA} python utils/stat_drift.py || true
done
# NOOP branch: raw vs auto-remap (phantom-motion check when holding still)
FR_NOOP=1 FR_RUN=pilot5_a8n_noop FR_CONFIG=configs/ar_eval_dmd_student.yaml \
  FR_CKPT=$CKPT FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
  python utils/flow_record_ode_student.py || echo "PROBE-FAIL noop"
ODE_EMDREMAP=auto:1 FR_NOOP=1 FR_RUN=pilot5_a8n_noop_auto FR_CONFIG=configs/ar_eval_dmd_student.yaml \
  FR_CKPT=$CKPT FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
  python utils/flow_record_ode_student.py || echo "PROBE-FAIL noopauto"
MC_RUNS=pilot5_a8n_noop:pilot5_a8n_noop_auto python utils/motion_check.py || true
echo "PROBE-A8N-AUTO done $(date)"
