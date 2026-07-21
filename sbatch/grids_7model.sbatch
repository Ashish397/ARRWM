#!/bin/bash
#SBATCH --job-name=grids-7model
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp MG_WORKERS=24
export MG_RUNS="pca8_8node,pca4,pca2,16node,4node,noatok,noadaln"
rm -f analysis/eval_final/grids/A/*_grid.mp4 analysis/eval_final/grids/B/*_grid.mp4
python utils/make_grids.py
echo "A: $(ls analysis/eval_final/grids/A/*_grid.mp4 | wc -l)/256  B: $(ls analysis/eval_final/grids/B/*_grid.mp4 | wc -l)/64"
echo "GRIDS-7MODEL done $(date)"
