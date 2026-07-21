#!/bin/bash
#SBATCH --job-name=world-replot
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Replot world groves from saved latent archives (adds grey seed points to the
# endpoints graphs). No re-encoding: npz reuse path.
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
for RUN in pca8_8node 16node minwm astra matrixgame worldcam worldplay yume; do
  echo "=== replot: $RUN ==="
  WG_RUN=$RUN python utils/world_grove.py || echo "REPLOT $RUN FAILED"
done
echo "WORLD-REPLOT done $(date)"
