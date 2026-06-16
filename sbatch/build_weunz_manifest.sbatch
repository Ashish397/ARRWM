#!/bin/bash
#SBATCH --job-name=build-weunz-manifest
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=02:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Build + cache the weunz (2639-ride) manifest. CPU/IO bound (reads each ride's
# caption embeds + zarr attrs). Reused by the scorer AND v14b training.
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
python utils/build_weunz_manifest.py
echo "Done $(date)"
