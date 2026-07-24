#!/bin/bash
#SBATCH --job-name=tb2_melt_vlm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/slurm/tb2_melt_vlm_%j.out
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
MELT_MODEL=qwen25vl7b  python analysis/testbench_v2/melt_vlm_bench.py
MELT_MODEL=internvl3_8b python analysis/testbench_v2/melt_vlm_bench.py
