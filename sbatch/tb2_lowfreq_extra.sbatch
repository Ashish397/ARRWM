#!/bin/bash
#SBATCH --job-name=tb2_lf_extra
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/tb2_lf_extra_%j.out

# Generalization probe set (r08/r01_R V1 mangle GT + external slice + real
# refs, ~73 videos): geometry qualifiers + PAL4VST/Qwen melt + stage-2
# features, sequentially on one GPU.
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TB2_CSD_CKPT=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache/hub/models--tomg-group-umd--CSD-ViT-L/snapshots/5bc26a6fb0487f3f00a2a7313135103a005b1b67/pytorch_model.bin
export TB2_EXTRA_ONLY=1

TB2_GEOM_OUT=analysis/testbench_v2/out/geometry_features_extra.csv \
  python analysis/testbench_v2/geometry_metrics.py
TB2_MANGLE_OUT=analysis/testbench_v2/out/mangle_features_extra.csv \
  python analysis/testbench_v2/mangle_metrics.py
TB2_GPU_OUT=analysis/testbench_v2/out/gpu_features_extra.csv \
  python analysis/testbench_v2/gpu_metrics.py
