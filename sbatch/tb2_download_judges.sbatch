#!/bin/bash
#SBATCH --job-name=tb2_dl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=08:00:00
#SBATCH --output=logs/slurm/tb2_dl_%j.out

# Judge checkpoint downloads (killed by login-node memory policing when run
# there): Qwen3-VL-30B-A3B (primary candidate) + InternVL3.5-38B-HF.
source /lus/lfs1aip2/scratch/u6ex/as1748.u6ex/miniforge3/bin/activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache

hf download Qwen/Qwen3-VL-30B-A3B-Instruct --max-workers 8
echo "=== Qwen3-VL-30B exit $? $(date) ==="
hf download OpenGVLab/InternVL3_5-38B-HF --max-workers 8
echo "=== InternVL3.5-38B exit $? $(date) ==="
