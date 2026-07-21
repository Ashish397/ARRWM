#!/bin/bash
#SBATCH --job-name=vlisa-bands
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party/VideoLISA
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
export VL_RUNS="pca8_8node:pca4:pca2:16node:4node:noatok:noadaln"
for TC in 0.20 0.25 0.30; do
  export VL_TOPCUT=$TC
  python videolisa_judge.py
  cp /scratch/u6ex/as1748.u6ex/ARRWM/analysis/eval_final/videolisa_scores.csv \
     /scratch/u6ex/as1748.u6ex/ARRWM/analysis/eval_final/videolisa_band${TC}.csv
  echo "BAND $TC done"
done
echo "VLISA-BANDS done"
