#!/bin/bash
#SBATCH --job-name=flow-rec-14es20
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Stage-1 of the ODE action-multiplicity campaign: 14e teacher chained rollout
# at TWENTY denoising steps (vs the 48-step trajs_14e8 reference), 8 blocks,
# window r08, 8 dirs x 4 seeds, pinned noise. Verifies the cheap-generation
# recipe (user est. ~40s/8-chunk window) before pilot LMDB generation:
# committed-chunk stats + endpoints must match 48-step within tolerance.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_s20 TRITON_CACHE_DIR=/tmp/tr_${USER}_s20

FR_RUN=14e8s20 FR_CONFIG=configs/causal_lora_diffusion_teacher_v14e.yaml \
  FR_CKPT=logs/v14e_pca8_raw/causal_lora_step0005000.pt \
  FR_STEPS=20 FR_BLOCKS=0,1,2,3,4,5,6,7 python utils/flow_record.py
echo "FLOW-REC-14ES20 done $(date)"
