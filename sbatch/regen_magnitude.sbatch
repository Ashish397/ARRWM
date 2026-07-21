#!/bin/bash
#SBATCH --job-name=regen-mag
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export MPLBACKEND=Agg PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
echo "=== magnitude following-by-REALdir (all runs) ==="
python utils/following_by_realdir.py
echo "=== response-gain (realized vs commanded, per axis) ==="
python utils/response_gain.py
echo "REGEN-MAGNITUDE done $(date)"
