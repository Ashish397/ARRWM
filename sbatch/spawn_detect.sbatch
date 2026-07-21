#!/bin/bash
#SBATCH --job-name=spawn-det
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --gpus-per-node=4
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# DINO patch spawn detector for all 13 phase-A models, 4 GPU shards on one node.
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
declare -A G=( [0]="pca8_8node:16node:pca4:pca2" [1]="4node:noatok:noadaln" \
               [2]="minwm:matrixgame:worldcam" [3]="yume:worldplay:astra" )
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g SD_RUNS="${G[$g]}" SD_OUT="analysis/eval_final/spawn_det_g${g}.csv" \
    python utils/spawn_detect.py > logs/spawn-det_${SLURM_JOB_ID}_g${g}.log 2>&1 &
done
wait
echo "SPAWN-DET done $(date)"
