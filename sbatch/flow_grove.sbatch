#!/bin/bash
#SBATCH --job-name=flow-grove
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# 3D grove-of-trees flow visualizations:
#  (1) flow_grove.py  - diffusion-time grove: 8 contexts x 1 noise x 8 actions,
#      new rollouts recording block-0 x_t (z = denoising step)
#  (2) world_grove.py - world-time grove: VAE re-encode the EXISTING 256 pca8
#      eval videos, z = video time; no new generation
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_fg TRITON_CACHE_DIR=/tmp/tr_${USER}_fg
python utils/world_grove.py
python utils/flow_grove.py
echo "FLOW-GROVE done $(date)"
