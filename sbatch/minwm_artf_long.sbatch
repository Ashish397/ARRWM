#!/bin/bash
#SBATCH --job-name=minwm-artf-long
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=4
#SBATCH --time=14:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party/minWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
export MW_STAGE=ar_tf MW_NUMLAT=36 MW_STEPS=48 MW_STEPT=0.32 MW_STEPR=3.0 MW_DIRS="F:FR:R:BR:B:BL:L:FL" MW_TAG=""
export MW_OUT=/scratch/u6ex/as1748.u6ex/ARRWM/logs/eval_final/A_minwm_artf
declare -A W=( [0]="0:1:2:3:4:5:6:7" [1]="8:9:10:11:12:13:14:15" [2]="16:17:18:19:20:21:22:23" [3]="24:25:26:27:28:29:30:31" )
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g MW_WINDOWS="${W[$g]}" python minwm_runner.py \
    > /scratch/u6ex/as1748.u6ex/ARRWM/logs/minwm-artf-long_${SLURM_JOB_ID}_g${g}.log 2>&1 &
done
wait
echo "videos: $(ls $MW_OUT/minwm_r*_*.mp4 2>/dev/null | wc -l)/256"
echo "MINWM-ARTF-LONG done $(date)"
