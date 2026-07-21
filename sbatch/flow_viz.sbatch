#!/bin/bash
#SBATCH --job-name=flow-viz
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Rectified-flow-style action-trajectory visualization (pca8): 8 dirs x 4 seeds,
# record block-0 x_t at all 48 denoise steps, shared PCA plot + separation curve.
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export TORCHINDUCTOR_CACHE_DIR=/tmp/ti_${USER}_fv TRITON_CACHE_DIR=/tmp/tr_${USER}_fv
python utils/flow_viz.py
echo "FLOW-VIZ done $(date)"
