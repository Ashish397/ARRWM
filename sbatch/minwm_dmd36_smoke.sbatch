#!/bin/bash
#SBATCH --job-name=minwm-dmd36
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party/minWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
# control: DMD student, SAME 141-frame length + 1x intensity as the artf test
export MW_STAGE=dmd MW_NUMLAT=36 MW_STEPT=0.32 MW_STEPR=3.0 \
  MW_WINDOWS="8:1" MW_DIRS="F:L" MW_TAG="_dmd36" \
  MW_OUT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/eval_final/minwm_smoke
python minwm_runner.py
echo "DMD36 done"
