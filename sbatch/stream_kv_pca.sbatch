#!/bin/bash
#SBATCH --job-name=stream-pca
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
export TMPDIR="/scratch/u6ex/as1748.u6ex/ARRWM/.tmp/${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/inductor"; mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" 2>/dev/null || true
export ARRWM_ACTION_ENCODER=pca
bash /scratch/u6ex/as1748.u6ex/ARRWM/logs/stream_kv_pca.sh
