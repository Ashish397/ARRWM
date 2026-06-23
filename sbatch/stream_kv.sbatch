#!/bin/bash
#SBATCH --job-name=stream-kv
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# BLOCK_CHUNKS / LOCAL_ATTN_CHUNKS / GEN_CHUNKS passed via --export at submit.
# Reset cache/tmp dirs: --export=ALL inherits the launching shell's TMPDIR /
# TORCHINDUCTOR_CACHE_DIR which may point at a harness-local path that doesn't
# exist on the compute node (PermissionError at torch.compile import).
export TMPDIR="/scratch/u6ex/as1748.u6ex/ARRWM/.tmp/${SLURM_JOB_ID}"
export TORCHINDUCTOR_CACHE_DIR="${TMPDIR}/inductor"
mkdir -p "${TORCHINDUCTOR_CACHE_DIR}" 2>/dev/null || true
bash /scratch/u6ex/as1748.u6ex/ARRWM/logs/stream_kv.sh
