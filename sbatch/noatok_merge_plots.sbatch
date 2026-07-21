#!/bin/bash
#SBATCH --job-name=noatok-merge
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Runs on a COMPUTE node (not the login node) after the ndof/iqa extractions.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export MPLBACKEND=Agg PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
python - <<'PY'
import os, pandas as pd
def merge(main, new, out):
    if not os.path.exists(new):
        print(f"[skip] {new} missing"); return
    m = pd.read_csv(main); n = pd.read_csv(new)
    m = m[m.run != "noatok"]
    pd.concat([m, n], ignore_index=True).to_csv(out, index=False)
    print(f"merged {new} -> {out}: noatok {len(n)} rows, steps {int(n.step.min())}..{int(n.step.max())}")
merge("analysis/iqa_14e_all.csv",   "analysis/iqa_noatok_full.csv",  "analysis/iqa_14e_all.csv")
merge("analysis/ndof_following.csv","analysis/ndof_noatok_full.csv", "analysis/ndof_following.csv")
PY
python utils/iqa_compare_groups.py
python utils/ndof_plots.py
echo "NOATOK-MERGE-PLOTS done $(date)"
