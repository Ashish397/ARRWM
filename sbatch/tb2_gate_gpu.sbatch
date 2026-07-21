#!/bin/bash
#SBATCH --job-name=tb2_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=02:00:00
#SBATCH --output=logs/slurm/tb2_gate_%j.out

# Testbench V2 GATE (b) run: stage-2 GPU features for ONLY the 84
# human-tier-labeled ablation videos (+ real siblings). No fleet-wide
# scoring happens until the tier-agreement report from this run is approved.
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TB2_CSD_CKPT=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache/hub/models--tomg-group-umd--CSD-ViT-L/snapshots/5bc26a6fb0487f3f00a2a7313135103a005b1b67/pytorch_model.bin

export TB2_LABELS_ONLY=1
export TB2_GPU_OUT=analysis/testbench_v2/out/gpu_features.csv
python analysis/testbench_v2/gpu_metrics.py
