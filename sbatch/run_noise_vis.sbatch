#!/bin/bash
#SBATCH --job-name=noise-vis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=1
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.out
#SBATCH --error=/scratch/u6ex/as1748.u6ex/ARRWM/logs/%x_%j.err

set -e -o pipefail

cd /scratch/u6ex/as1748.u6ex/ARRWM

source /scratch/u6ex/as1748.u6ex/miniforge3/bin/activate
conda activate arrwm

export HF_HOME=/scratch/u6ex/as1748.u6ex/frodobots/hf_cache
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== Step 1: Secant decomposition + save vis bundles ==="
python noise_analysis_backbone.py --mode secant --save_vis 5 --max_windows 20 --gamma 0.5 --device cuda:0

echo "=== Step 2: Decode and render vis bundles ==="
python vis_noise_secant.py --bundle_dir vis/noise_analysis/vis_bundles --video --device cuda:0

echo "=== Done ==="
ls -lh vis/noise_analysis/renders/
