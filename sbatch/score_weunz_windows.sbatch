#!/bin/bash
#SBATCH --job-name=score-weunz-windows
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --exclusive
#SBATCH --time=03:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Score EVERY block-aligned window of the weunz (2639-ride) manifest by
# motion + backward-ness, on GPU, sharded across 32 ranks (8 nodes x 4).
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
CACHE_DIR='/scratch/u6ex/as1748.u6ex/frodobots/hf_cache'
export HF_HOME=$CACHE_DIR HF_HUB_CACHE=$CACHE_DIR HUGGINGFACE_HUB_CACHE=$CACHE_DIR TRANSFORMERS_CACHE=$CACHE_DIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8
mkdir -p paper_assets/window_scores_weunz

srun --ntasks=32 --ntasks-per-node=4 --gpus-per-node=4 bash -c '
  export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
  python utils/score_all_windows.py \
    --manifest logs/v14b_weunz/.ride_manifest.pt \
    --outdir paper_assets/window_scores_weunz
'
echo "=== shards written ==="
ls -la paper_assets/window_scores_weunz/
echo "Done $(date)"
