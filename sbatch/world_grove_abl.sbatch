#!/bin/bash
#SBATCH --job-name=grove-abl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# World-time groves for the remaining 5 ablations (pca8_8node + 16node already done).
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
for RUN in pca4 pca2 4node noatok noadaln; do
  echo "=== world grove: $RUN ==="
  WG_RUN=$RUN python utils/world_grove.py || echo "GROVE $RUN FAILED"
done
echo "GROVE-ABL done $(date)"
