#!/bin/bash
#SBATCH --job-name=tb2_gpu_recap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm/tb2_gpu_recap_%j.out

# Re-run stage-2 features on the probe set under the 6s-generation cap
# (previous gpu_features_extra.csv evaluated externals to native length).
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TB2_CSD_CKPT=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache/hub/models--tomg-group-umd--CSD-ViT-L/snapshots/5bc26a6fb0487f3f00a2a7313135103a005b1b67/pytorch_model.bin
export TB2_EXTRA_ONLY=1

TB2_GPU_OUT=analysis/testbench_v2/out/gpu_features_extra.csv \
  python analysis/testbench_v2/gpu_metrics.py
