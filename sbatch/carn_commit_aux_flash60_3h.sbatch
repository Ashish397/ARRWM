#!/bin/bash
#SBATCH --job-name=carn-commit-aux-flash60
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

# Direct three-hour batch allocation for the matched Flash-era-routing
# follow-up. The shared launcher creates one two-node torchrun step inside
# this allocation and preserves the campaign's 300/15/200 cadence.
set -euo pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

HOLDER="$SLURM_JOB_ID" \
CARNVX_MODE=commit_aux_flash60 \
PORTOFF=27307 \
MAXSTEPS=300 \
RUNSTAMP=flashrestore2708 \
bash sbatch/run_carn_vx_on_holder.sh
