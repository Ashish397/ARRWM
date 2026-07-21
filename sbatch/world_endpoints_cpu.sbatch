#!/bin/bash
#SBATCH --job-name=ep-replot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Endpoints-only world-grove replot (adds grey seed points). CPU-only: no GPU
# request -> schedules fast; reads npz archives + PCA + one PNG per model.
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
for RUN in pca8_8node 16node minwm astra matrixgame worldcam worldplay yume; do
  WG_RUN=$RUN WG_ENDPOINTS_ONLY=1 python utils/world_grove.py || echo "EP $RUN FAILED"
done
echo "EP-REPLOT done $(date)"
