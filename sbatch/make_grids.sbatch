#!/bin/bash
#SBATCH --job-name=make-grids
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --time=01:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export MG_WORKERS=24
python utils/make_grids.py
echo "=== grid counts ==="
echo "A: $(ls analysis/eval_final/grids/A/*_grid.mp4 2>/dev/null | wc -l)/256   B: $(ls analysis/eval_final/grids/B/*_grid.mp4 2>/dev/null | wc -l)/64"
echo "MAKE-GRIDS done $(date)"
