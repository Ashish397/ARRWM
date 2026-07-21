#!/bin/bash
#SBATCH --job-name=vlm-bench
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=05:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
VB_INTERNVL=/scratch/u6ex/as1748.u6ex/ARRWM/third_party/_vlm/InternVL3-8B VB_SA2VA=/scratch/u6ex/as1748.u6ex/ARRWM/third_party/_vlm/Sa2VA-4B VB_JUDGES="${VB_JUDGES:-sa2va:internvl3}" ${VB_PYTHON:-python} utils/vlm_bench.py
echo "VLM-BENCH done"
