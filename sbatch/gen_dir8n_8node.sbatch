#!/bin/bash
#SBATCH --job-name=gen-dir8n
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --time=03:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# dir8n LMDB: 8 compass directions + NO-OP per context (9 chains/context),
# balanced 6000-window pool interleaved over 32 shards, 40 contexts/shard
# -> up to 1280 contexts / ~11.5k chains (~26s/chain, fits 3h).
# Resumable: atomic per-chain writes, existing files skipped.
srun bash /scratch/u6ex/as1748.u6ex/ARRWM/sbatch/_gen_dir8n_node.sh
echo "GEN-DIR8N done $(date)"
