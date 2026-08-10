#!/bin/bash
#SBATCH --job-name=gen-dir8n-ns1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# SECOND noise realization of the dir8n chains (GL_NOISE_VARIANT=1): same
# contexts, same 8 actions + no-op, DIFFERENT teacher noise. Required for
# CONDITIONAL distribution matching — with one chain per (context, action)
# there is no empirical p(z_next | C, a) for the student to match.
export GL_NV=1
srun --export=ALL bash /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/_gen_dir8n_node.sh
echo "GEN-DIR8N-NS1 done $(date)"
