#!/bin/bash
#SBATCH --job-name=tb2_j_q30b
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/tb2_j_q30b_%j.out

# Gate (c) bake-off, candidate 1: Qwen3-VL-30B-A3B (MoE) on the 215 labeled
# cross-tier pairs (both orders judged inside the script = 430 forward
# passes with ~27 images each). bf16, device_map=auto over 2 GPUs.
# Number to beat: composite LOSO 0.737.
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TB2_JUDGE=Qwen/Qwen3-VL-30B-A3B-Instruct

python analysis/testbench_v2/vlm_pairwise.py --mode pair \
  --pairs analysis/testbench_v2/out/gate_pairs.csv \
  --out analysis/testbench_v2/out/gate_vlm_qwen3vl30b.csv
