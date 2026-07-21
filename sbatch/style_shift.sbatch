#!/bin/bash
#SBATCH --job-name=style-shift
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Style-shift localization (VGG-Gram + CLIP zero-shot) on the matrixgame
# browse video. CPU-only analysis; the login node kills the VGG pass.
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
python utils/style_shift_detect.py \
  analysis/eval_final/browse_baseline_smoke/matrixgame_r01_R.mp4
echo "STYLE-SHIFT done $(date)"
