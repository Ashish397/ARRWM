#!/bin/bash
#SBATCH --job-name=tb2_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=logs/slurm/tb2_cpu_%j.out

# Testbench V2 stage 1: CPU-only pixel/haze/murk features over the whole fleet.
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache

# TB2_MODELS="" -> all A_* fleets + real refs
export TB2_CPU_OUT=analysis/testbench_v2/out/cpu_features.csv
export TB2_CURVES_DIR=analysis/testbench_v2/out/curves

python analysis/testbench_v2/cpu_metrics.py
