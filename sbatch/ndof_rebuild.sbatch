#!/bin/bash
#SBATCH --job-name=ndof-rebuild
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --gpus-per-node=4
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Re-extract the 5 token runs' N-DOF (clobbered by a hardcoded-NDOF_OUT bug),
# merge back with noatok, and regenerate the gain / magnitude / roll-pitch plots.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export MPLBACKEND=Agg HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH

# 1) preserve current noatok rows (already full 100-5000)
python -c "import pandas as pd; d=pd.read_csv('analysis/ndof_following.csv'); n=d[d.run=='noatok']; n.to_csv('analysis/ndof_noatok_keep.csv',index=False); print('kept noatok rows:',len(n))"

# 2) re-extract the 5 token runs, split across the 4 GPUs
declare -A RUNS=( [0]="pca8_8node" [1]="16node" [2]="4node,pca4" [3]="pca2" )
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g NDOF_RUNS="${RUNS[$g]}" NDOF_STRIDE=200 NDOF_OUT="analysis/ndof_g${g}.csv" \
    TORCHINDUCTOR_CACHE_DIR="/tmp/ti_${USER}_$g" TRITON_CACHE_DIR="/tmp/tr_${USER}_$g" \
    python utils/ndof_following.py > "logs/ndof-rebuild_${SLURM_JOB_ID}_gpu${g}.log" 2>&1 &
done
wait

# 3) merge the 5 token CSVs + kept noatok back into the canonical CSV
python -c "
import pandas as pd, glob
parts=[pd.read_csv(f) for f in sorted(glob.glob('analysis/ndof_g*.csv'))] + [pd.read_csv('analysis/ndof_noatok_keep.csv')]
d=pd.concat(parts, ignore_index=True); d.to_csv('analysis/ndof_following.csv', index=False)
print('rebuilt ndof_following.csv runs:', sorted(d.run.unique()), 'rows', len(d))
"

# 4) regenerate the plots
python utils/response_gain.py
python utils/following_by_realdir.py
python utils/ndof_plots.py
python utils/circular_error.py
echo "NDOF-REBUILD done $(date)"
