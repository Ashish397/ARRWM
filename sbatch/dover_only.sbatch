#!/bin/bash
#SBATCH --job-name=dover-only
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party/DOVER
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export TMPDIR=/tmp
for br in BL B; do
  for run in pca8_8node pca4 pca2 16node 4node noatok; do
    v=/scratch/u6ex/as1748.u6ex/ARRWM/logs/eval_final/A/${run}/control_test/step05000_r08_${br}_raw.mp4
    echo "##### DOVER ${br} ${run} #####"
    python evaluate_one_video.py -v "$v" 2>&1 | tail -6
  done
done
echo "DOVER-ONLY done $(date)"
