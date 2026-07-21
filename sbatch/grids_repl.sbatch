#!/bin/bash
#SBATCH --job-name=grids-repl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=00:40:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp MG_WORKERS=24
for wi in 00 01 04 07 10 18 21 24 27; do rm -f analysis/eval_final/grids/A/r${wi}_*_grid.mp4; done
for wi in 42 50 56 63; do rm -f analysis/eval_final/grids/B/r${wi}_static_grid.mp4; done
python utils/make_grids.py
echo "A: $(ls analysis/eval_final/grids/A/*_grid.mp4 | wc -l)/256  B: $(ls analysis/eval_final/grids/B/*_grid.mp4 | wc -l)/64"
echo "GRIDS-REPL done $(date)"
