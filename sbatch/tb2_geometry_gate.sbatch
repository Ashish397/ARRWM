#!/bin/bash
#SBATCH --job-name=tb2_geom
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/tb2_geom_%j.out

# Testbench V2 stage 2.6 (geometry qualifiers) on the 84 labeled videos:
# DepthAnything-V2 depth-field roughness/instability + LSD line-segment stats.
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TB2_LABELS_ONLY=1
python analysis/testbench_v2/geometry_metrics.py
