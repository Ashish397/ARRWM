#!/bin/bash
#SBATCH --job-name=carn-commit-aux-txboth
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --open-mode=append
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

# Full CARN permutation requested on 2026-08-27:
#   tx +former and -latter + aux-minus + committed-memory correction.
# The shared launcher pins the restored Flash-era domains and the campaign's
# 300-step / sample-15 / checkpoint-200 contract as last-wins overrides.
set -euo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

HOLDER="$SLURM_JOB_ID" \
CARNVX_MODE=commit_aux_tx_both \
PORTOFF=27308 \
MAXSTEPS=300 \
RUNSTAMP=flashrestore2708 \
bash sbatch/run_carn_vx_snapshot.sh
