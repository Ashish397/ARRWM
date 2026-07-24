#!/bin/bash
#SBATCH --job-name=tb2_boundary
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/slurm/tb2_boundary_%j.out
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TB2_CSD_CKPT=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache/hub/models--tomg-group-umd--CSD-ViT-L/snapshots/5bc26a6fb0487f3f00a2a7313135103a005b1b67/pytorch_model.bin
export TB2_LABELS_ONLY=1 TB2_CORRECT_BOUNDARY=1
TB2_MANGLE_OUT=analysis/testbench_v2/out/mangle_features_corrected.csv python analysis/testbench_v2/mangle_metrics.py
TB2_GEOM_OUT=analysis/testbench_v2/out/geometry_features_corrected.csv python analysis/testbench_v2/geometry_metrics.py
