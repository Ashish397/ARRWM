#!/bin/bash
#SBATCH --job-name=vlisa-comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# VideoLISA melt-band judge on comparator eval videos (all 8 dirs x 32 windows).
# Usage: sbatch --export=ALL,VL_RUNS=minwm:astra,VL_OUT=...csv sbatch/vlisa_comp.sbatch
set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM/third_party/VideoLISA
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
unset PYTHONPATH
export VL_ALLDIRS=1 VL_TOPCUT=0.25
python videolisa_judge.py
echo "VLISA-COMP done $(date)"
