#!/bin/bash
#SBATCH --job-name=tb2_judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=12:00:00
#SBATCH --output=logs/slurm/tb2_judge_%j.out

# Testbench V2 stage 3: VLM pairwise judge (V2 upgrade 1 = 30B-72B bf16).
# A 72B judge in bf16 (~145GB) shards across all 4 GPUs via device_map=auto,
# so run ONE process (no sharding loop). For a 7B judge you can instead run
# 4 shards like stage 2.
#
# Prerequisites:
#   1. bradley_terry.py schedule  (writes out/judge_pairs.csv + judge_auto.csv)
#   2. judge weights present in $HF_HOME (download once from a login node:
#      huggingface-cli download Qwen/Qwen2.5-VL-72B-Instruct)
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
cd /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM
export PYTHONPATH=/lus/lfs1aip2/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export TB2_JUDGE=${TB2_JUDGE:-Qwen/Qwen2.5-VL-72B-Instruct}

# content pairs (prescreened by the composite)
python analysis/testbench_v2/vlm_pairwise.py --mode pair \
  --pairs analysis/testbench_v2/out/judge_pairs.csv \
  --out analysis/testbench_v2/out/vlm_pair.csv

# style question (instrument c) over the whole fleet incl. real refs
python analysis/testbench_v2/vlm_pairwise.py --mode style \
  --out analysis/testbench_v2/out/vlm_style.csv
