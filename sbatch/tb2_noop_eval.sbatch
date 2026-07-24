#!/bin/bash
#SBATCH --job-name=tb2_noop
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm/tb2_noop_%j.out

# No-op (null-action) reference-based eval: decode real continuations for
# phase-B (64) + phase-A (32) windows via WanVAE, then paired
# PSNR/SSIM/LPIPS + FVD per model (7 ablations x 64 static rollouts +
# matrixgame_nullact x 64 vs A refs).
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python analysis/testbench_v2/noop_reference_eval.py all
