#!/bin/bash
#SBATCH --job-name=cm-noadaln
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
CM_RUNS="noadaln" CM_STRIDE=200 CM_OUT="analysis/chunk_metrics_noadaln.csv" \
  TORCHINDUCTOR_CACHE_DIR="/tmp/ti_${USER}_na" TRITON_CACHE_DIR="/tmp/tr_${USER}_na" \
  python utils/chunk_metrics.py
python -c "
import pandas as pd
base = pd.read_csv('analysis/chunk_metrics.csv')
na = pd.read_csv('analysis/chunk_metrics_noadaln.csv')
assert 'noadaln' not in set(base.run), 'noadaln already merged'
d = pd.concat([base, na], ignore_index=True)
d.to_csv('analysis/chunk_metrics.csv', index=False)
print('merged: runs', sorted(d.run.unique()), 'rows', len(d))
"
echo "CM-NOADALN done $(date)"
