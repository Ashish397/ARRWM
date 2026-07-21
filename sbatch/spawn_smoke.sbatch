#!/bin/bash
#SBATCH --job-name=spawn-smoke
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err
# Spawn-detector smoke on the 6 by-eye-verified minwm videos.
set -o pipefail
cd /scratch/u6ex/as1748.u6ex/ARRWM
source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate; conda activate arrwm
export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache PYTHONPATH=/scratch/u6ex/as1748.u6ex/ARRWM:$PYTHONPATH TMPDIR=/tmp
SD_RUNS=minwm SD_ONLY="r02_FR,r02_R,r02_BR,r02_L,r06_B,r06_F" \
  SD_OUT=analysis/eval_final/spawn_smoke.csv \
  SD_NPZ_DIR=analysis/eval_final/spawn_masks_smoke \
  python utils/spawn_detect.py
cat analysis/eval_final/spawn_smoke.csv
echo "SPAWN-SMOKE done $(date)"
