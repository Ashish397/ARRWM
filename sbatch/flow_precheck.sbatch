#!/bin/bash
#SBATCH --job-name=flow-precheck
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Pre-checks before any Fix-1/Fix-2 training change (window r08):
#   A) DETERMINISTIC-CHAINING re-measurement: same students, same pinned
#      noise, but rungs chained with the block's ORIGINAL noise (straight
#      rectified-flow path) instead of fresh re-noise. If rung convergence
#      contracts vs the stochastic recordings, Problem 1 is an inference-
#      sampler fix, no training needed.
#   B) REALIZED-MOTION videos: 8-chunk rollouts per dir (2 seeds), decoded
#      to mp4, for the commanded-vs-realized egomotion check (is the mode-
#      misassignment real motion error or metric geometry?).
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_fpc TRITON_CACHE_DIR=/tmp/tr_${USER}_fpc

P3=logs/action_forcing_phase3_dmd_one_step
declare -A SCKPT=(
  [afall_freal_cd]=$P3/af_all_freal_cd_j5313179/phase1_step0000500.pt
  [statwave_Freal]=$P3/stat_wave_dmd3_fwd_Freal_j5302282/phase1_step0000200.pt
)

for run in afall_freal_cd statwave_Freal; do
  echo "=== A: deterministic-chain recording $run ==="
  FR_RUN=$run FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=${SCKPT[$run]} \
    FR_TAG=_det FR_DET=1 python utils/flow_record_ode_student.py

  echo "=== B: motion-check videos $run ==="
  FR_RUN=$run FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=${SCKPT[$run]} \
    FR_TAG=_vid FR_VIDEO=1 FR_CHUNKS=8 FR_NSEEDS=2 python utils/flow_record_ode_student.py
done
echo "FLOW-PRECHECK done $(date)"
