#!/bin/bash
#SBATCH --job-name=build-curated-pool
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Merge weunz window-score shards -> NMS -> most-motion-y pool + backward
# oversampler -> paper_assets/v14b_train_windows.json. CPU/JSON only.
set -u -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm
python utils/build_curated_pool.py \
  --shards_dir paper_assets/window_scores_weunz \
  --out paper_assets/v14b_train_windows.json
echo "Done $(date)"
