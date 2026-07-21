#!/bin/bash
#SBATCH --job-name=chunk-metrics
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
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
# the 6 runs split across the 4 GPUs (noatok has the most videos -> its own GPU)
declare -A RUNS=( [0]="noatok" [1]="16node" [2]="pca8_8node,pca4" [3]="pca2,4node" )
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g CM_RUNS="${RUNS[$g]}" CM_STRIDE=200 CM_OUT="analysis/chunk_metrics_g${g}.csv" \
    TORCHINDUCTOR_CACHE_DIR="/tmp/ti_${USER}_$g" TRITON_CACHE_DIR="/tmp/tr_${USER}_$g" \
    python utils/chunk_metrics.py > "logs/chunk-metrics_${SLURM_JOB_ID}_gpu${g}.log" 2>&1 &
done
wait
python -c "
import pandas as pd, glob
parts=[pd.read_csv(f) for f in sorted(glob.glob('analysis/chunk_metrics_g*.csv'))]
d=pd.concat(parts, ignore_index=True); d.to_csv('analysis/chunk_metrics.csv', index=False)
print('merged chunk_metrics.csv: runs', sorted(d.run.unique()), 'rows', len(d),
      '| has_iqa', int(d.musiq.notna().sum()), '| has_recon(GT)', int(d.mae.notna().sum()))
"
echo "CHUNK-METRICS done $(date)"
