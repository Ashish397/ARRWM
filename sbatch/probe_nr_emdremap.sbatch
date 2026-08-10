#!/bin/bash
#SBATCH --job-name=probe-emdremap
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Solution 3 closed-form: serve-time EMD quantile remap toward the seed
# context's per-channel quantile profile, on the nr checkpoint (the
# serve-series base: comparable to lock/inv/hyb/chroma/repnudge).
# Two blend strengths: lam=0.5 and lam=1.0.
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=$PWD:$PWD/action-forcing TMPDIR=/tmp
CKPT=logs/ode14e_pilot/run3_flip2_nr/action_ode_step0000250.pt
for LAM in 0.5 1.0; do
  TAG=$(echo $LAM | tr -d .)
  ODE_EMDREMAP=$LAM FR_RUN=pilot3_nr_emdremap${TAG} FR_CONFIG=configs/ar_eval_dmd_student.yaml \
    FR_CKPT=$CKPT FR_RUNGS="1000,625,357.142857,208.333333" FR_VIDEO=1 FR_CHUNKS=6 FR_NSEEDS=2 \
    python utils/flow_record_ode_student.py || echo "PROBE-FAIL emdremap$TAG"
  FDD_NSEEDS=2 FDD_PAIRS="pilot3_nr_emdremap${TAG}=14e8" FDD_FIGNAME=flow_pilot3_nr_emdremap${TAG}_scorecard \
    python utils/flow_diverge_dmd3.py || echo "SCORE-FAIL emdremap$TAG"
  MC_RUNS=pilot3_nr_emdremap${TAG} python utils/motion_check.py || true
  SD_RUNS=pilot3_flip2nr:pilot3_nr_emdremap${TAG} python utils/stat_drift.py || true
done
echo "PROBE-EMDREMAP done $(date)"
