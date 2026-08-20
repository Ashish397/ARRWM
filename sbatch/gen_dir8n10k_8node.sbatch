#!/bin/bash
#SBATCH --job-name=gen-dir8n10k
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# dir8n 10K-pool expansion: ~8,766 new contexts x 9 chains @ ~26s/chain
# = ~18h of work over 32 GPUs. Submitted as a chain of three 8h instances
# (afterany) — each resumes exactly where the previous stopped (per-file
# existence skip), shorter jobs land faster than one 24h block.
srun bash /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/_gen_dir8n10k_node.sh
echo "GEN-DIR8N10K instance done $(date)"
