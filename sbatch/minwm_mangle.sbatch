#!/bin/bash
#SBATCH --job-name=minwm-mangle
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export MC_RUNS="pca8_8node:pca4:pca2:16node:4node:noatok:minwm"
# note: minwm dirs are label-swapped for L/R families; B and F unaffected
for W in "8 B" "8 BL" "1 R" "2 F"; do
  set -- $W; MC_RANK=$1 MC_BRANCH=$2 python utils/mangle_composite.py
done
echo "MINWM-MANGLE done"
