#!/bin/bash
#SBATCH --job-name=minwm-artf
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party/minWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
if [ ! -f ckpts/Wan21/Action2V/ar_diffusion_tf/model.pt ]; then
  hf download MIN-Lab/minWM --local-dir ./ckpts --include "Wan21/Action2V/ar_diffusion_tf/*" 2>&1 | tail -1
fi
export MW_STAGE=ar_tf MW_STEPT=0.32 MW_STEPR=3.0 MW_WINDOWS="8" MW_DIRS="F:B:L:R" MW_TAG="_artf" \
  MW_OUT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/eval_final/minwm_smoke
python minwm_runner.py
echo "MINWM-ARTF-SMOKE done"
