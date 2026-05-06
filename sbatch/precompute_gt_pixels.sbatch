#!/bin/bash
#SBATCH --job-name=precompute-gt-pixels
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM

# Capture + clear positional args before sourcing conda's activate script
# (miniforge's activate errors on >1 positional arg when our forwarded
# n_rides arg leaks into it).
_orig_args=("$@")
set --
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
set -- "${_orig_args[@]}"

CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR
export HF_HUB_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8

# Reuse v6's manifest (the smoke uses the same one — ensures ride order
# matches the smoke's iteration so the first N rides decoded here are
# the most-likely ones to be hit in a 20-step smoke).
MANIFEST=/scratch/u6ex/as1748.u6ex/ARRWM/logs/action_forcing_phase1_gan_v6/.ride_manifest.pt
OUT=/scratch/u6ex/as1748.u6ex/ARRWM/gt_pixels_cache
N_RIDES=${1:-15}

echo "manifest=$MANIFEST"
echo "out=$OUT"
echo "n_rides=$N_RIDES"
mkdir -p "$OUT"

srun python bin/precompute_gt_pixels.py \
    --manifest "$MANIFEST" \
    --out "$OUT" \
    --n_rides "$N_RIDES" \
    --decode_chunk 21
