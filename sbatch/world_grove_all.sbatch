#!/bin/bash
#SBATCH --job-name=world-grove
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# (1) patch 8 missing head-to-head motion rows, then (2) world-time groves for
# 16node + all comparators (pca8_8node already done). VAE re-encode, no new gen.
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
python utils/hh_patch_missing.py || echo "HH-PATCH FAILED"
for RUN in 16node minwm astra matrixgame worldcam worldplay yume; do
  echo "=== world grove: $RUN ==="
  WG_RUN=$RUN python utils/world_grove.py || echo "GROVE $RUN FAILED"
done
echo "WORLD-GROVE-ALL done $(date)"
