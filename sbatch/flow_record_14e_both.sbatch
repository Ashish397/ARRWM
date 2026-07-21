#!/bin/bash
#SBATCH --job-name=flow-rec-14eb
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=07:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Resubmission of 5736570 + 5736740 (hung on a shared-.rec_scratch manifest
# write collision; flow_record.py now uses per-run scratch): 14e teacher
# 8 blocks at 48 steps AND at 20 steps, sequentially in one job.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_14eb TRITON_CACHE_DIR=/tmp/tr_${USER}_14eb

CFG=configs/causal_lora_diffusion_teacher_v14e.yaml
CKPT=logs/v14e_pca8_raw/causal_lora_step0005000.pt

echo "=== 14e8: 48-step x 8 blocks ==="
FR_RUN=14e8 FR_CONFIG=$CFG FR_CKPT=$CKPT \
  FR_BLOCKS=0,1,2,3,4,5,6,7 python utils/flow_record.py

echo "=== 14e8s20: 20-step x 8 blocks ==="
FR_RUN=14e8s20 FR_CONFIG=$CFG FR_CKPT=$CKPT \
  FR_STEPS=20 FR_BLOCKS=0,1,2,3,4,5,6,7 python utils/flow_record.py
echo "FLOW-REC-14EB done $(date)"
