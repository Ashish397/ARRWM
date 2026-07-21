#!/bin/bash
#SBATCH --job-name=tb2_mangle
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/tb2_mangle_%j.out

# Testbench V2 stage 2.5 (low-frequency collapse) on the 84 labeled videos:
# PAL4VST artifact fraction + Qwen2.5-VL-7B melt-question P(Yes).
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TB2_LABELS_ONLY=1
python analysis/testbench_v2/mangle_metrics.py
