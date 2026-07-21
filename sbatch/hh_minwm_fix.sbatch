#!/bin/bash
#SBATCH --job-name=hh-mwfix
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --gpus-per-node=1
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp ARRWM_ACTION_ENCODER=pca_raw
HH_MODELS="minwm" HH_OUT="analysis/eval_final/hh_motion_minwm.csv" python utils/headtohead_extract.py
python -c "
import pandas as pd, glob
parts=[pd.read_csv(f) for f in sorted(glob.glob('analysis/eval_final/hh_motion_g*.csv'))]
d=pd.concat(parts, ignore_index=True)
d=d[d.model!='minwm']
mw=pd.read_csv('analysis/eval_final/hh_motion_minwm.csv')
d=pd.concat([d,mw], ignore_index=True)
d.to_csv('analysis/eval_final/headtohead_motion.csv', index=False)
print('remerged', len(d), '| minwm g0 non-null:', d[d.model=='minwm'].g0.notna().sum())"
echo "HH-MWFIX done"
