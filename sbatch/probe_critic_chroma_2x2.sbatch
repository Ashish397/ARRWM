#!/bin/bash
#SBATCH --job-name=probe-cc2x2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# 2x2: {nr retrain, nr+boosted action critic} x {chroma-lock, hybrid} serve.
# Answers: (a) does chroma-only locking keep lock's colour stability without
# its action kill; (b) does the action critic harden actions under locking.
set -x
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp

probe () {  # $1 arm dir suffix, $2 run base, $3 lock mode, $4 tag
  CKPT=logs/ode14e_pilot/run3_flip2_$1/action_ode_step0000250.pt
  [ -f "$CKPT" ] || { echo "SKIP $1 (no ckpt)"; return; }
  ODE_STAT_LOCK=$3 FR_RUN=$2 FR_TAG=$4 FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$CKPT \
    FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || echo "PROBE-FAIL $2$4"
  FDD_NSEEDS=2 FDD_PAIRS="$2$4=14e8" FDD_FIGNAME=flow_$2${4}_scorecard \
    python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL $2$4"
  MC_RUNS=$2$4 python utils/motion_check.py || true
  SD_RUNS=$2$4 python utils/stat_drift.py || true
}
probe nr   pilot4_nr   chroma _chroma
probe nr   pilot4_nr   hyb    _hyb
probe nrcg pilot4_nrcg chroma _chroma
probe nrcg pilot4_nrcg hyb    _hyb
echo "CC2X2 done $(date)"
