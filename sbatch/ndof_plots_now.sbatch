#!/bin/bash
#SBATCH --job-name=ndof-plots
#SBATCH -N 1 --ntasks-per-node=1 --cpus-per-task=32 --gpus-per-node=1 --time=00:20:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail; cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export MPLBACKEND=Agg PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
python - <<'PY'
import os,pandas as pd
n="analysis/ndof_noatok_full.csv"
if os.path.exists(n):
    m=pd.read_csv("analysis/ndof_following.csv"); nn=pd.read_csv(n)
    m=m[m.run!="noatok"]; pd.concat([m,nn],ignore_index=True).to_csv("analysis/ndof_following.csv",index=False)
    print(f"ndof merged: noatok {len(nn)} rows steps {int(nn.step.min())}..{int(nn.step.max())}")
PY
python utils/response_gain.py
python utils/following_by_realdir.py
python utils/ndof_plots.py
echo "NDOF-PLOTS-NOW done $(date)"
