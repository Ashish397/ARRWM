#!/bin/bash
#SBATCH --job-name=iqa-noatok-regen
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export MPLBACKEND=Agg TMPDIR=/tmp HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER} TRITON_CACHE_DIR=/tmp/tr_${USER}
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
# 1) IQA on noatok's full video set (now in the canonical dir)
export IQA_RUNS=noatok IQA_OUT=analysis/iqa_noatok_full.csv
python utils/iqa_control_videos.py
# 2) merge noatok into the master IQA csv (replace old sparse rows)
python -c "
import pandas as pd
m=pd.read_csv('analysis/iqa_14e_all.csv'); n=pd.read_csv('analysis/iqa_noatok_full.csv')
m=m[m.run!='noatok']; pd.concat([m,n],ignore_index=True).to_csv('analysis/iqa_14e_all.csv',index=False)
print('IQA merged: noatok now',len(n),'rows steps',int(n.step.min()),'..',int(n.step.max()))
"
# 3) regenerate the action-tokens (and other) comparison graphs
python utils/iqa_compare_groups.py
echo "IQA-NOATOK-REGEN done $(date)"
