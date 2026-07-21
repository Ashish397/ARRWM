#!/bin/bash
#SBATCH --job-name=suite-comp
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# VLM suite (PAL + Qwen P(Yes) + DINO spawn) on comparator eval videos.
# Usage: sbatch --export=ALL,SA_RUNS=minwm:astra,SA_OUT=...csv sbatch/suite_comp.sbatch
set -e
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache TMPDIR=/tmp
PYTHONPATH=$PWD python utils/suite_scoreA.py
echo "SUITE-COMP done $(date)"
