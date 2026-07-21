#!/bin/bash
#SBATCH --job-name=flow-rec-afdeg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# AR-degradation flow recordings (window r08, pinned noise):
#   1) 14d teacher dense over EIGHT blocks (production local-attn window) ->
#      matched reference for the students' 8-chunk committed-stat drift
#   2) the two remaining well-trained af_* students (no recordings yet):
#      af_roll_wave_j5273323@400, af_roll_wave_dmd3_j5285487@400
#      - standard 2-chunk x 4 seeds (rung/convergence analyses)
#      - 8-chunk x 2 seeds "_vid" (degradation stats + realized-motion videos)
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_fad TRITON_CACHE_DIR=/tmp/tr_${USER}_fad

echo "=== recording 14d teacher blocks 0..7 ==="
FR_RUN=14d8 FR_CONFIG=configs/causal_lora_diffusion_teacher_v14d.yaml \
  FR_CKPT=logs/v14d_critic8_natural/causal_lora_step0002150.pt \
  FR_BLOCKS=0,1,2,3,4,5,6,7 python utils/flow_record.py

P3=logs/action_forcing_phase3_dmd_one_step
declare -A SCKPT=(
  [afroll_wave]=$P3/af_roll_wave_j5273323/phase1_step0000400.pt
  [afroll_wave_dmd3]=$P3/af_roll_wave_dmd3_j5285487/phase1_step0000400.pt
)
for run in afroll_wave afroll_wave_dmd3; do
  echo "=== recording student $run (2-chunk x4 seeds) ==="
  FR_RUN=$run FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=${SCKPT[$run]} \
    python utils/flow_record_ode_student.py
  echo "=== recording student $run (8-chunk x2 seeds + videos) ==="
  FR_RUN=$run FR_CONFIG=configs/ar_eval_dmd_student.yaml FR_CKPT=${SCKPT[$run]} \
    FR_TAG=_vid FR_VIDEO=1 FR_CHUNKS=8 FR_NSEEDS=2 python utils/flow_record_ode_student.py
done
echo "FLOW-REC-AFDEG done $(date)"
