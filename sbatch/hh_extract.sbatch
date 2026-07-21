#!/bin/bash
#SBATCH --job-name=hh-extract
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --gpus-per-node=4
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp ARRWM_ACTION_ENCODER=pca_raw
declare -A G=( [0]="minwm:pca8_8node" [1]="pca4:pca2" [2]="16node:4node" [3]="noatok:noadaln" )
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g HH_MODELS="${G[$g]}" HH_OUT="analysis/eval_final/hh_motion_g${g}.csv" \
    TORCHINDUCTOR_CACHE_DIR=/tmp/ti_$g python utils/headtohead_extract.py \
    > logs/hh-extract_${SLURM_JOB_ID}_g${g}.log 2>&1 &
done
wait
python -c "
import pandas as pd, glob
d=pd.concat([pd.read_csv(f) for f in sorted(glob.glob('analysis/eval_final/hh_motion_g*.csv'))], ignore_index=True)
d.to_csv('analysis/eval_final/headtohead_motion.csv', index=False)
print('merged', len(d), 'rows | models:', sorted(d.model.unique()))"
echo "HH-EXTRACT done $(date)"
