#!/bin/bash
#SBATCH --job-name=minwm-speed
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party/minWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
export MW_STAGE=ar_tf MW_STEPT=0.32 MW_STEPR=3.0 MW_WINDOWS="8" MW_DIRS="F:B" \
  MW_OUT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/eval_final/minwm_smoke
echo "--- variant A: steps=24, CFG=3.0 ---"
MW_STEPS=24 MW_GUIDE=3.0 MW_TAG="_s24g3" python minwm_runner.py
echo "--- variant B: steps=48, CFG off ---"
MW_STEPS=48 MW_GUIDE=1.0 MW_TAG="_s48g1" python minwm_runner.py
echo "MINWM-SPEED done"
