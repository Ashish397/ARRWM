#!/bin/bash
#SBATCH --job-name=nc-reloc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Scene-relocation (sibling consensus) for the No Critic variant. CPU-only:
# ORB/RANSAC over end frames of the full fleet. Heavy sequential video decode,
# so it runs off the login node.
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
export OPENBLAS_NUM_THREADS=8 OMP_NUM_THREADS=8
python grids/eval/nocritic_reloc.py full
echo "NC-RELOC done $(date)"
