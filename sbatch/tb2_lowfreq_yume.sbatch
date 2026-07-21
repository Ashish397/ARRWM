#!/bin/bash
#SBATCH --job-name=tb2_lf_yume
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/tb2_lf_yume_%j.out

# Supplement to the fleet job (which excluded yume while it regenerated):
# low-freq ensemble ingredients for the fresh A_yume (256 videos), capped.
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TB2_MODELS=yume

TB2_MANGLE_OUT=analysis/testbench_v2/out/fleet_mangle_yume.csv \
  python analysis/testbench_v2/mangle_metrics.py
TB2_GEOM_OUT=analysis/testbench_v2/out/fleet_geometry_yume.csv \
  python analysis/testbench_v2/geometry_metrics.py
