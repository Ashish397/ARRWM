#!/bin/bash
#SBATCH --job-name=probe-repnudge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Serve-time STAT LOCK test (theoretically motivated by the variance
# contraction law): re-probe the arm-D (nr) checkpoint with ODE_STAT_LOCK=repnudge
# (per-channel affine re-pin of each committed chunk to the real seed
# context's stats, applied before emission + KV refresh). Zero training.
# Compare pilot3_nrmom2_repnudge vs pilot3_flip2nr: drift should go ~1.0;
# actions should be UNTOUCHED (the renorm is content-preserving affine).
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
export ODE_STAT_LOCK=repnudge

CKPT=logs/ode14e_pilot/run3_flip2_nrmom2/action_ode_step0000250.pt
FR_RUN=pilot3_nrmom2 FR_TAG=_repnudge FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=$CKPT \
  FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
  python utils/flow_record_ode_student.py || echo "PROBE-FAIL repnudge"
FDD_NSEEDS=2 FDD_PAIRS="pilot3_nrmom2_repnudge=14e8" FDD_FIGNAME=flow_pilot3_nrmom2_repnudge_scorecard \
  python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL repnudge"
MC_RUNS=pilot3_nrmom2_repnudge python utils/motion_check.py || true
SD_RUNS=pilot3_flip2nr:pilot3_nrmom2_repnudge python utils/stat_drift.py || true
echo "PROBE-REPNUDGE done $(date)"
